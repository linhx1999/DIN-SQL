"""Offline regression checks for completion-based resume and isolated logs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_bird_dev as runner


class FakePipeline:
    def run_question(self, index, row, database_path, chat):
        print(f"unique-question-{index}")
        if index == 1:
            raise ValueError("deliberate failure")
        return {"final_query": f"SELECT {index}"}


def initialize_fake_worker():
    runner._PIPELINE = FakePipeline()
    runner._CHAT = object()


class RunnerTests(unittest.TestCase):
    def test_parallel_logs_failures_and_export_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            rows = [{"db_id": "db", "question": f"Q{i}"} for i in range(3)]
            with ProcessPoolExecutor(
                max_workers=2,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_fake_worker,
            ) as pool:
                futures = [
                    pool.submit(
                        runner.process_question,
                        i,
                        row,
                        {"data_root": directory},
                        directory,
                    )
                    for i, row in enumerate(rows)
                ]
                records = [future.result() for future in reversed(futures)]
            self.assertEqual(records[1]["status"], "failed")
            self.assertEqual(records[1]["final_query"], "")
            for index in range(3):
                log = (root / "logs" / f"question_{index}.log").read_text()
                self.assertIn(f"unique-question-{index}", log)
                for other in set(range(3)) - {index}:
                    self.assertNotIn(f"unique-question-{other}", log)
            runner.export_results(root, rows, {r["index"]: r for r in records})
            predictions = json.loads((root / f"{root.name}.json").read_text())
            self.assertEqual(list(predictions), ["0", "1", "2"])
            self.assertEqual(predictions["1"], "\t----- bird -----\tdb")

    def test_resume_skips_success_and_failure_and_checks_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "run"
            output.mkdir()
            database = root / "dev_databases" / "db"
            database.mkdir(parents=True)
            (database / "db.sqlite").touch()
            rows = [{"db_id": "db", "question": f"Q{i}"} for i in range(2)]
            (root / "dev.json").write_text(json.dumps(rows))
            args = argparse.Namespace(data_root=root, max_tokens=12, workers=2)
            # Setup configuration without initiating inference.
            with patch.object(
                runner, "load_pipeline", side_effect=RuntimeError("setup")
            ):
                with self.assertRaisesRegex(RuntimeError, "setup"):
                    runner.run(args, output)
            records = [
                dict(
                    index=i,
                    db_id="db",
                    final_query="SELECT 0" if i == 0 else "",
                    status="success" if i == 0 else "failed",
                )
                for i in range(2)
            ]
            checkpoint = output / "output.jsonl"
            checkpoint.write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            with patch.object(
                runner, "load_pipeline", side_effect=AssertionError("must skip")
            ):
                args.workers = 4
                self.assertEqual(runner.run(args, output), 1)
            self.assertEqual(len(checkpoint.read_text().splitlines()), 2)
            args.max_tokens = 13
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                runner.run(args, output)
            args.max_tokens = 12
            rows[0]["question"] = "changed"
            (root / "dev.json").write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                runner.run(args, output)

    def test_interrupted_tail_is_unfinished_and_other_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.jsonl"
            rows = [{"db_id": "db"}, {"db_id": "db"}]
            first = (
                json.dumps(dict(index=0, db_id="db", status="failed", final_query=""))
                + "\n"
            )
            path.write_text(first + '{"index":1')
            completed = runner.read_completed(path, rows)
            self.assertEqual(list(completed), [0])
            self.assertEqual(path.read_text(), first)
            self.assertEqual(
                path.with_suffix(".jsonl.partial").read_text(), '{"index":1'
            )
            path.write_text(first + "broken\n")
            with self.assertRaisesRegex(ValueError, "Invalid checkpoint"):
                runner.read_completed(path, rows)


if __name__ == "__main__":
    unittest.main()
