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
                chat = MagicMock()
                chat.predict.side_effect = ['Schema_links: [t.id]', f'Label: "{label}" questions = ["sub"]', 'SQL: SELECT id FROM t', 'SELECT id FROM t']
                with patch.object(pipeline, 'load_prompts', return_value=prompts):
                    result = pipeline.run_question(0, {'question': 'Q', 'db_id': 'db'}, '/data/database', chat)
                self.assertEqual(result['final_query'], 'SELECT id FROM t')
                self.assertEqual(chat.predict.call_count, 4)
                self.assertEqual([c.args[0] for c in chat.client.set_stage.call_args_list], ['schema_linking', 'classification', 'sql_generation', 'self_correction'])
        self.assertEqual(pipeline.extract_sql('id FROM t', True), 'SELECT id FROM t')

    def test_ordered_export_keeps_missing_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p/'run_config.json').write_text(json.dumps({'dataset': 'spider'}))
            runner.export_results(p, [{'db_id': 'db'}, {'db_id': 'db'}], {1: {'final_query': 'SELECT 1\nFROM t'}})
            self.assertEqual((p/'pred_dev.sql').read_text().splitlines(), ['SELECT * FROM __din_failed_prediction__', 'SELECT 1 FROM t'])
