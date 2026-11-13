"""Offline request accounting and resume-session telemetry checks."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from generation_log import LoggedClient, append_record


def write_one(path, index):
    append_record(Path(path), {'idx': index})


class GenerationLogTests(unittest.TestCase):
    def test_retry_stage_usage_and_response_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            response = {'choices': [{'finish_reason': 'length'}], 'usage': {'prompt_tokens': 12, 'completion_tokens': 7}}
            client = Mock()
            client.create.side_effect = [RuntimeError('secret'), response, {'choices': []}]
            wrapped = LoggedClient(client, directory, {'idx': 2, 'execution_id': 'e'})
            wrapped.set_stage('schema_linking')
            with self.assertRaises(RuntimeError):
                wrapped.create(model='m', messages=[])
            self.assertIs(wrapped.create(model='m', messages=[]), response)
            wrapped.set_stage('classification')
            wrapped.create(model='m', messages=[])
            text = (Path(directory) / 'llm_attempts.jsonl').read_text()
            self.assertNotIn('secret', text)
            rows = list(map(json.loads, text.splitlines()))
            self.assertEqual([r['attempt'] for r in rows], [1, 2, 1])
            self.assertEqual(rows[0]['call_id'], rows[1]['call_id'])
            self.assertNotEqual(rows[1]['call_id'], rows[2]['call_id'])
            self.assertIsNone(rows[0]['prompt_tokens'])
            self.assertEqual(rows[1]['completion_tokens'], 7)
            self.assertEqual(rows[1]['finish_reason'], 'length')
            self.assertIsNone(rows[2]['completion_tokens'])

    def test_process_writes_are_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'llm_attempts.jsonl')
            with ProcessPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(write_one, path, i) for i in range(60)]
                for future in futures:
                    future.result()
            rows = list(map(json.loads, Path(path).read_text().splitlines()))
            self.assertEqual({r['idx'] for r in rows}, set(range(60)))

    def test_worker_wraps_and_restores_client_on_failure(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import run_bird_dev as runner
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'logs').mkdir()
            client = Mock()
            client.create.return_value = {'usage': {'prompt_tokens': 1, 'completion_tokens': 2}}
            chat = SimpleNamespace(client=client)
            def question(index, row, path, chat):
                chat.client.set_stage('classification')
                chat.client.create(model='m', messages=[])
                raise ValueError('private detail')
            with patch.object(runner, '_PIPELINE', SimpleNamespace(run_question=question)), patch.object(runner, '_CHAT', chat):
                result = runner.process_question(0, {'db_id': 'db', 'question': 'q'}, {'data_root': directory}, directory)
            self.assertIs(chat.client, client)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['error'], 'ValueError')
            self.assertGreaterEqual(result['exec_time'], 0)
            record = json.loads((Path(directory) / 'llm_attempts.jsonl').read_text())
            self.assertEqual(record['execution_id'], result['execution_id'])
            self.assertEqual(record['stage'], 'classification')
