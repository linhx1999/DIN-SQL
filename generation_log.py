"""Process-safe request telemetry for the legacy LangChain/OpenAI client."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid


def append_record(path: Path, record: dict) -> None:
    with path.open('a', encoding='utf-8') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def value(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


class LoggedClient:
    def __init__(self, client, output_dir: str, context: dict):
        self.client = client
        self.path = Path(output_dir) / 'llm_attempts.jsonl'
        self.context = context
        self.set_stage('unknown')

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.call_id = uuid.uuid4().hex
        self.attempt = 0

    def __getattr__(self, key):
        return getattr(self.client, key)

    def create(self, **kwargs):
        self.attempt += 1
        started = time.monotonic()
        record = dict(self.context, stage=self.stage, call_id=self.call_id,
                      attempt=self.attempt, request_scope='sdk_invocation',
                      model=kwargs.get('model'), started_unix=time.time(),
                      prompt_tokens=None, completion_tokens=None, reasoning_tokens=None,
                      prompt_sha256=hashlib.sha256(json.dumps(kwargs.get('messages'), sort_keys=True).encode()).hexdigest())
        try:
            response = self.client.create(**kwargs)
            usage = value(response, 'usage')
            choices = value(response, 'choices') or []
            record.update(status='success', response_received=True,
                          prompt_tokens=value(usage, 'prompt_tokens'),
                          completion_tokens=value(usage, 'completion_tokens'),
                          reasoning_tokens=value(value(usage, 'completion_tokens_details'), 'reasoning_tokens'),
                          finish_reason=value(choices[0], 'finish_reason') if choices else None)
            return response
        except Exception as exc:
            record.update(status='failed', exception_class=type(exc).__name__)
            raise
        finally:
            record['elapsed_seconds'] = time.monotonic() - started
            append_record(self.path, record)
