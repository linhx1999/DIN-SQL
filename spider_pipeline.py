"""Spider DIN-SQL stages using the original prompt builders."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import re

_PROMPTS = None
_SCHEMA_PATH = None


def load_prompts(schema_path: Path):
    global _PROMPTS, _SCHEMA_PATH
    if _PROMPTS is None:
        spec = importlib.util.spec_from_file_location('din_spider_prompts', Path(__file__).with_name('DIN-SQL.py'))
        _PROMPTS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_PROMPTS)
    if _SCHEMA_PATH != schema_path:
        _PROMPTS.spider_schema, _PROMPTS.spider_primary, _PROMPTS.spider_foreign = _PROMPTS.creatiing_schema(str(schema_path))
        _SCHEMA_PATH = schema_path
    return _PROMPTS


class ResponseFormatError(ValueError):
    """The model did not return a usable stage answer."""


def extract_sql(text: str, select_prefix: bool = False) -> str:
    # Chat models must return a complete query, not a completion after SELECT.
    from sql_response import extract_query
    query = extract_query(text, revised=True)
    if not query:
        raise ResponseFormatError("Expected a complete SQL query")
    return query


def parse_classification(text: str) -> tuple[str, str]:
    import ast
    # Prefer an explicit final label; tolerate markdown, quotes and casing.
    labels = re.findall(r"(?:label|classification)\s*[*_:：\s]*[\"'`*]*(NON[- _]NESTED|NESTED|EASY)\b", text, re.I)
    if not labels:
        labels = re.findall(r"(?:classified\s+as|class\s+is)\s*[\"'`*]*(NON[- _]NESTED|NESTED|EASY)\b", text, re.I)
    if not labels:
        bare = re.fullmatch(r"\s*[\"'`*]*(NON[- _]NESTED|NESTED|EASY)[\"'`*.\s]*", text, re.I)
        labels = [bare.group(1)] if bare else []
    if not labels:
        raise ResponseFormatError("Missing or unrecognized classification label")
    label = re.sub(r"[- _]", "-", labels[-1].upper())
    sub_questions = ""
    matches = list(re.finditer(r"(?:sub[-_ ]?)?questions\s*[*_\s]*[=:：]\s*(\[[\s\S]*?\])", text, re.I))
    if matches:
        try:
            values = ast.literal_eval(matches[-1].group(1))
            if isinstance(values, list) and all(isinstance(v, str) for v in values):
                sub_questions = " ; ".join(values)
        except (ValueError, SyntaxError):
            pass
    return label, sub_questions


def validate_sql(query: str, database: Path) -> None:
    import sqlite3
    with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("EXPLAIN " + query).fetchall()


def run_question(index: int, row: dict, dev_db_path: str, chat) -> dict:
    prompts = load_prompts(Path(dev_db_path).parent / 'tables.json')
    question, db = row['question'], row['db_id']
    record = {}
    stage = 'schema_linking'

    def ask(name, prompt, stops=None):
        nonlocal stage
        stage = name
        client = getattr(chat, 'client', None)
        if hasattr(client, 'set_stage'):
            client.set_stage(name)
        result = chat.predict(prompt, stop=stops)
        record[name] = result
        if not isinstance(result, str) or not result.strip():
            raise ResponseFormatError('Empty model response')
        return result

    sql_contract = "\nReturn exactly one complete SQLite SELECT or WITH query in a ```sql``` block. Do not return an explanation or a fragment after SELECT."
    try:
        linking = ask('schema_linking', prompts.schema_linking_prompt_maker(question, db))
        match = re.search(r'Schema[_ ]links\s*:\s*(\[[\s\S]*?\])', linking, re.I)
        links = match.group(1) if match else '[]'
        classification = ask('classification', prompts.classification_prompt_maker(question, db, links) +
                             '\nEnd with Label: EASY, NON-NESTED, or NESTED. For NESTED also return Sub_questions: ["question", ...].')
        label, sub_questions = parse_classification(classification)
        record['parsed_classification'] = label
        if label == 'EASY':
            prompt = prompts.easy_prompt_maker(question, db, links)
        elif label == 'NON-NESTED':
            prompt = prompts.medium_prompt_maker(question, db, links)
        else:
            record['sub_questions_missing'] = not bool(sub_questions)
            prompt = prompts.hard_prompt_maker(question, db, links, sub_questions)
        generated = ask('sql_generation', prompt + sql_contract)
        sql = extract_sql(generated)
        # Remove the legacy completion prefix. Preserve the original instructions.
        correction_prompt = re.sub(r'\nSELECT\s*$', '', prompts.debuger(question, db, sql))
        corrected = ask('self_correction', correction_prompt + sql_contract +
                        '\nIf no correction is needed, return the original complete SQL query.')
        final = extract_sql(corrected)
        stage = 'final_validation'
        validate_sql(final, Path(dev_db_path) / db / (db + '.sqlite'))
        record['final_query'] = final
        return record
    except Exception as exc:
        record['failure_stage'] = stage
        exc.partial_result = record
        raise
