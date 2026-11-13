# Few-shot-NL2SQL-with-prompting

## Table of contents
* [Dataset](#dataset)
* [Setup](#setup)
* [Citation](#citation)


## dataset
To reproduce the results reported in the paper, please download the Spider dataset from the link below and create a data directory containing the tables.json and dev.json files.

```
$ Spider dataset = "https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ"
```


## setup
To run this project, use the following commands:

```
$ pip3 install -r requirements.txt
$ echo "Start running DIN-SQL.py"
$ python3 DIN-SQL.py --dataset ./data/ --output predicted_sql.txt
$ echo "Finished running DIN-SQL.py"
```
## BIRD dev

Use Python 3.10 and install `requirements.txt`. The BIRD pipeline uses the
legacy LangChain/OpenAI interfaces pinned there.

```bash
bash run_bird_dev.sh
# Arguments override the wrapper's defaults:
bash run_bird_dev.sh --workers 4 --output_dir outputs/my_run
```

Edit the explicit configuration in `run_bird_dev.sh` for the API endpoint,
model, token budget, dataset and run name. Defaults match the MAG-SQL wrapper:
local API at `http://127.0.0.1:8000/v1`, model
`Ministral-3-14B-Reasoning-2512`, 16384 max tokens, 8 worker processes, and
`../SAGE-SQL/datasets/bird/dev`. The default run name is
`dinsql_bird_dev_Ministral-3-14B-Reasoning-2512_run_1`.

Input layout:

```text
<data_root>/dev.json
<data_root>/dev_databases/<db_id>/<db_id>.sqlite
<data_root>/dev_databases/<db_id>/database_description/*.csv  # optional
```

Each question uses schema linking, classification, SQL generation and
self-correction. Generation and correction prompts request the final query in
a single Markdown code block tagged `sql`, allowing normal multiline formatting.
Extraction prefers the final SQL code block and also accepts legacy `SQL:` /
`Revised_SQL:` prefixes and bare SELECT/WITH queries. It removes comments and
normalizes whitespace outside
quoted literals and identifiers without joining adjacent SQL tokens. Gold SQL
and difficulty are excluded from worker inputs. The main process writes shared predictions and checkpoints; workers append request telemetry under a process lock:

- `<output_dir>/<output_dir_name>.json`: BIRD predictions keyed by original
  zero-based dataset position, with `SQL\t----- bird -----\tdb_id` values.
  Failed and unfinished questions retain their positions with empty SQL.
- `output.jsonl`: durable completion records, including stage outputs or errors.
- `logs/question_<index>.log`: per-question stage output and exception classes.
- `logs.csv`: aggregate completed-question records (without gold SQL).
- `run_config.json`: dataset path/dev.json SHA-256, model, endpoint and token budget.
  API keys are not stored.

Run the same command to resume. **Every saved completion is skipped, whether
successful or failed.** Only unfinished questions run again. Completion means a
record has been flushed to `output.jsonl`; a worker result lost before that point
will be rerun. A torn final append is saved as `output.jsonl.partial` and removed
on resume; other checkpoint corruption raises an error. Outputs are rebuilt
from completion records, and predictions remain in input order.

Resume rejects changed dataset paths/dev.json contents, models, endpoints or
token budgets; use a new output directory for a different experiment. Worker
count may change. Keep SQLite files and description CSVs unchanged within a run
(they are not content-hashed). A directory lock prevents simultaneous writers.
Single-question exceptions are recorded and processing continues; the command
exits with status 1 if any completed question failed. Such failures are still
skipped on subsequent resumes. Process crashes without completion records remain
unfinished. No benchmark evaluation or automatic retry of completed failures is
performed.

After changing prompts or extraction code, use a fresh `--output_dir` to rerun
previously failed questions and avoid mixing implementations within one run.

Runner and SQL extraction checks (no model endpoint needed):

```bash
python -m unittest discover -s tests -v
bash -n run_bird_dev.sh
```

## citation 

``` 
@article{pourreza2023din,
  title={DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction},
  author={Pourreza, Mohammadreza and Rafiei, Davood},
  journal={arXiv preprint arXiv:2304.11015},
  year={2023}
}
```


### Generation cost telemetry

`run_bird_dev.py` now records generation cost automatically; the existing shell
launcher needs no additional flag. Use a new `RUN_NAME` / output directory for
a complete cost experiment. Resuming an old run still skips all saved completions:
missing historical cost data cannot be reconstructed by resuming.

- `llm_attempts.jsonl`: process-locked, flushed request records with `idx`,
  session/execution IDs, stage, call ID, attempt number, model, input/output/
  reasoning tokens, elapsed time, finish reason, and exception class. Stages
  distinguish schema linking, classification, SQL generation (three branches),
  and self-correction. Request text, API keys, and raw exception messages are
  not included. A request success means a response was returned, not correct SQL.
- `output.jsonl` and `logs.csv`: add `exec_time` and `execution_id`; failures
  also retain elapsed time. Existing prediction format and completion-based
  resume rules remain unchanged. Exception details are reduced to class names
  in per-question logs and results to avoid exposing provider-echoed credentials.
- `run_sessions.jsonl`: append one record per invocation, including code hashes,
  worker count, completion state, and duration. `run_time.json` holds only the
  latest invocation, not the sum across resumes. Python `finally` records failed
  runs too, but SIGKILL/machine failure can prevent session finalization.

Calls are measured at the legacy SDK `create` boundary. LangChain retry attempts
crossing this boundary are visible; any internal SDK retries are not individually
observable. The original retry settings, prompts, and token limits are unchanged.
Missing token usage is null, never zero. Reasoning tokens may be included in
completion tokens and must not simply be added to them. Per-question time includes
worker setup, schema processing and retry waiting, but excludes executor queueing;
request time and run wall time are separate measures, neither representing GPU time.
Preserve all request records when a crashed question is retried: `execution_id`
distinguishes those costs from the final saved completion. Do not infer complete
cost coverage from complete predictions if the run began before telemetry existed.

Offline verification: `python -m unittest discover -s tests -v` (mock requests;
no inference or benchmark evaluation).
