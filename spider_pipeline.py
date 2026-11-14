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


def extract_sql(text: str, select_prefix: bool = False) -> str:
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:sql)?\s*|\s*```$', '', text, flags=re.I).strip()
    if 'SQL:' in text:
        text = text.rsplit('SQL:', 1)[1].strip()
    if select_prefix and not re.match(r'^(SELECT|WITH)\b', text, re.I):
        text = 'SELECT ' + text
    if not text:
        raise ValueError('Empty SQL response')
    return text


def run_question(index: int, row: dict, dev_db_path: str, chat) -> dict:
    prompts = load_prompts(Path(dev_db_path).parent / 'tables.json')
    question, db = row['question'], row['db_id']

    def ask(stage, prompt, stops):
        client = getattr(chat, 'client', None)
        if hasattr(client, 'set_stage'):
            client.set_stage(stage)
        result = chat.predict(prompt, stop=stops)
        if not isinstance(result, str) or not result.strip():
            raise ValueError('Empty model response')
        return result

    linking = ask('schema_linking', prompts.schema_linking_prompt_maker(question, db), ['Q:'])
    links = linking.split('Schema_links: ', 1)[1] if 'Schema_links: ' in linking else '[]'
    classification = ask('classification', prompts.classification_prompt_maker(question, db, links[1:]), ['Q:'])
    label = classification.split('Label: ', 1)[1] if 'Label: ' in classification else '"NESTED"'
    if '"EASY"' in label:
        prompt = prompts.easy_prompt_maker(question, db, links)
    elif '"NON-NESTED"' in label:
        prompt = prompts.medium_prompt_maker(question, db, links)
    else:
        sub_questions = classification.split('questions = ["', 1)[1].split('"]', 1)[0]
        prompt = prompts.hard_prompt_maker(question, db, links, sub_questions)
    generated = ask('sql_generation', prompt, ['Q:'])
    sql = extract_sql(generated)
    corrected = ask('self_correction', prompts.debuger(question, db, sql), ['#', ';', '\n\n'])
    return {'final_query': extract_sql(corrected, select_prefix=True),
            'schema_linking': linking, 'classification': classification,
            'sql_generation': generated, 'self_correction': corrected}
