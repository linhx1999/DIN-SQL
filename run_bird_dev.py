"""Parallel BIRD runner with durable, completion-based resume."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import csv
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import traceback

_PIPELINE = None
_CHAT = None


def load_pipeline():
    spec = importlib.util.spec_from_file_location(
        "din_sql_bird", Path(__file__).with_name("DIN-SQL_BIRD.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_question(index: int, row: dict, config: dict, output_dir: str) -> dict:
    """Keep clients and stdout isolated in each worker process."""
    global _PIPELINE, _CHAT
    result = {"index": index, "db_id": row["db_id"], "question": row["question"]}
    log_path = Path(output_dir) / "logs" / f"question_{index}.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            print(f"Starting question {index}", flush=True)
            try:
                if _PIPELINE is None:
                    _PIPELINE = load_pipeline()
                if _CHAT is None:
                    from langchain.chat_models import ChatOpenAI

                    _CHAT = ChatOpenAI(
                        model=config["model"],
                        temperature=0,
                        max_tokens=config["max_tokens"],
                        openai_api_base=config["api_base"],
                    )
                result.update(
                    _PIPELINE.run_question(
                        index,
                        row,
                        str(Path(config["data_root"]) / "dev_databases"),
                        _CHAT,
                    )
                )
                result["status"] = "success"
            except Exception as exc:
                traceback.print_exc()
                result.update(status="failed", final_query="", error=str(exc))
            print(f"Completed question {index}: {result['status']}", flush=True)
    return result


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_completed(path: Path, rows: list) -> dict:
    """Recover only an interrupted final append; reject other corruption."""
    completed = {}
    if not path.exists():
        return completed
    with path.open("rb+") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                # Preserve the incomplete bytes before removing the torn append.
                path.with_suffix(".jsonl.partial").write_bytes(line)
                stream.truncate(offset)
                print("Recovered an interrupted checkpoint append.")
                break
            try:
                record = json.loads(line)
                index = record["index"]
                if (
                    type(index) is not int
                    or not 0 <= index < len(rows)
                    or index in completed
                    or record["db_id"] != rows[index]["db_id"]
                    or record["status"] not in {"success", "failed"}
                    or not isinstance(record["final_query"], str)
                ):
                    raise ValueError("Invalid completion record")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid checkpoint at byte {offset}: {path}"
                ) from exc
            completed[index] = record
    return completed


def export_results(output_dir: Path, rows: list, completed: dict) -> None:
    predictions = {
        str(
            index
        ): f"{completed.get(index, {}).get('final_query', '')}\t----- bird -----\t{row['db_id']}"
        for index, row in enumerate(rows)
    }
    atomic_json(output_dir / f"{output_dir.name}.json", predictions)
    columns = [
        "index",
        "question",
        "db_id",
        "status",
        "final_query",
        "schema_linking",
        "classification",
        "sql_generation",
        "self_correction",
        "error",
    ]
    temporary = output_dir / "logs.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(completed[index] for index in sorted(completed))
    temporary.replace(output_dir / "logs.csv")


def run(args: argparse.Namespace, output_dir: Path) -> int:
    data_root = args.data_root.resolve()
    dataset = (data_root / "dev.json").read_bytes()
    data = json.loads(dataset)
    # Gold SQL and difficulty never enter worker state or prompts.
    rows = [
        {
            "db_id": row["db_id"],
            "question": row["question"],
            "evidence": row.get("evidence", ""),
        }
        for row in data
    ]
    for db_id in {row["db_id"] for row in rows}:
        database = data_root / "dev_databases" / db_id / f"{db_id}.sqlite"
        if not database.is_file():
            raise FileNotFoundError(database)
    config = {
        "format_version": 1,
        "data_root": str(data_root),
        "dev_json_sha256": hashlib.sha256(dataset).hexdigest(),
        "model": os.environ.get("MODEL_NAME", "gpt-4-32k"),
        "api_base": os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        "max_tokens": args.max_tokens,
    }
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError(
                "Run configuration changed; use a different output_dir / RUN_NAME"
            )
    else:
        if (output_dir / "output.jsonl").exists():
            raise ValueError("Checkpoint exists without run_config.json")
        atomic_json(config_path, config)
    (output_dir / "logs").mkdir(exist_ok=True)
    checkpoint = output_dir / "output.jsonl"
    completed = read_completed(checkpoint, rows)
    export_results(output_dir, rows, completed)
    pending = [(index, row) for index, row in enumerate(rows) if index not in completed]
    print(
        f"Total: {len(rows)}; completed: {len(completed)}; pending: {len(pending)}",
        flush=True,
    )
    if pending:
        # Fail setup before marking any questions completed when dependencies are missing.
        load_pipeline()
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY (use EMPTY for a local endpoint)")
        with checkpoint.open("a", encoding="utf-8") as stream:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                futures = {
                    executor.submit(
                        process_question, index, row, config, str(output_dir)
                    ): index
                    for index, row in pending
                }
                for future in as_completed(futures):
                    # Worker/process crashes remain incomplete and retryable on resume.
                    record = future.result()
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    completed[record["index"]] = record
                    export_results(output_dir, rows, completed)
                    print(
                        f"[{len(completed)}/{len(rows)}] question {record['index']}: {record['status']}",
                        flush=True,
                    )
    failures = sum(record["status"] == "failed" for record in completed.values())
    print(
        f"Finished: {len(completed)} completed, {failures} failed. Output: {output_dir}"
    )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=Path("dev"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--max_tokens", type=int, default=int(os.environ.get("MAX_TOKENS", "2000"))
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.max_tokens < 1:
        parser.error("workers and max_tokens must be positive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Concurrent invocations must not append to the same checkpoint.
    with (output_dir / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another process is already using this output_dir")
        return run(args, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
