from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import run_bird_dev as runner
import spider_pipeline as pipeline


class SpiderPipelineTest(unittest.TestCase):
    def test_all_generation_branches_and_prefix(self):
        for label in ['EASY', 'NON-NESTED', 'NESTED']:
            with self.subTest(label=label):
                prompts = MagicMock()
                prompts.debuger.return_value = "Fix SQL\nSELECT"
                chat = MagicMock()
                chat.predict.side_effect = ['Schema_links: [t.id]', f'Label: "{label}" questions = ["sub"]', 'SQL: SELECT id FROM t', 'SELECT id FROM t']
                with patch.object(pipeline, 'load_prompts', return_value=prompts), patch.object(pipeline, 'validate_sql'):
                    result = pipeline.run_question(0, {'question': 'Q', 'db_id': 'db'}, '/data/database', chat)
                self.assertEqual(result['final_query'], 'SELECT id FROM t')
                self.assertEqual(chat.predict.call_count, 4)
                self.assertEqual([c.args[0] for c in chat.client.set_stage.call_args_list], ['schema_linking', 'classification', 'sql_generation', 'self_correction'])
        with self.assertRaises(pipeline.ResponseFormatError):
            pipeline.extract_sql('id FROM t', True)

    def test_ordered_export_keeps_missing_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'run_config.json').write_text(json.dumps({'dataset': 'spider'}))
            runner.export_results(p, [{'db_id': 'db'}, {'db_id': 'db'}], {1: {'final_query': 'SELECT 1\nFROM t'}})
            self.assertEqual((p/'pred_dev.sql').read_text().splitlines(), ['SELECT * FROM __din_failed_prediction__', 'SELECT 1 FROM t'])

    def test_classification_response_variants(self):
        examples = [
            ('**Label:** NON-NESTED', ('NON-NESTED', '')),
            ('label: easy', ('EASY', '')),
            ('NESTED', ('NESTED', '')),
            ('Label: "NESTED"\nSub_questions: ["one", "two"]', ('NESTED', 'one ; two')),
            ("Label: 'NESTED'\nquestions = ['one']", ('NESTED', 'one')),
            ('The SQL query can be classified as "EASY".', ('EASY', '')),
        ]
        for text, expected in examples:
            with self.subTest(text=text):
                self.assertEqual(pipeline.parse_classification(text), expected)
        with self.assertRaises(pipeline.ResponseFormatError):
            pipeline.parse_classification('No usable answer')

    def test_real_observed_correction_prose_is_rejected(self):
        text = 'The given SQLite SQL QUERY is already correct for the question "How many singers do we have?" since it simply counts all rows in the `singer` table.'
        with self.assertRaises(pipeline.ResponseFormatError):
            pipeline.extract_sql(text, True)
        self.assertEqual(pipeline.extract_sql(text + '\n```sql\nSELECT COUNT(*) FROM singer;\n```'), 'SELECT COUNT(*) FROM singer;')

    def test_stops_removed_and_failure_retains_stage_outputs(self):
        prompts = MagicMock()
        prompts.debuger.return_value = 'Instructions\nSELECT'
        chat = MagicMock()
        chat.predict.side_effect = ['Schema_links: [t.id]', '**Label:** NON-NESTED', 'SELECT id FROM t', 'The query is already correct.']
        with patch.object(pipeline, 'load_prompts', return_value=prompts):
            with self.assertRaises(pipeline.ResponseFormatError) as caught:
                pipeline.run_question(0, {'question': 'Q', 'db_id': 'db'}, '/data/database', chat)
        record = caught.exception.partial_result
        self.assertEqual(record['failure_stage'], 'self_correction')
        self.assertEqual(record['sql_generation'], 'SELECT id FROM t')
        self.assertTrue(all(c.kwargs['stop'] is None for c in chat.predict.call_args_list))
        self.assertNotIn('Instructions\nSELECT', chat.predict.call_args_list[-1].args[0])

    def test_sql_validation_rejects_prose_and_keeps_database_unchanged(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp)/'fixture.sqlite'
            with sqlite3.connect(db) as conn:
                conn.execute('CREATE TABLE t (id INTEGER)')
            pipeline.validate_sql('SELECT id AS my_id FROM t', db)
            with self.assertRaises(sqlite3.Error):
                pipeline.validate_sql('SELECT The given query is already correct', db)
