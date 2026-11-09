"""Offline regression tests for model SQL response formats."""

import sqlite3
import unittest

from sql_response import extract_query


class SQLResponseTests(unittest.TestCase):
    def test_formats_and_trailing_explanation(self):
        for response in (
            'Reasoning\nSQL: SELECT\n1;\nExplanation follows.',
            'Here is the query:\n```sql\nSELECT\n1;\n```\nExplanation.',
            '```SQL\nSELECT 1\n```',
            'SELECT 1;',
            'sql: SELECT 1;',
            '**SQL:** SELECT 1;',
        ):
            with self.subTest(response=response):
                query = extract_query(response)
                self.assertIsNotNone(query)
                with sqlite3.connect(':memory:') as db:
                    self.assertEqual(db.execute(query).fetchall(), [(1,)])

    def test_final_code_block_takes_priority_over_legacy_label(self):
        response = 'SQL: SELECT 0;\nFinal answer:\n```sql\nSELECT\n2;\n```'
        for revised in (False, True):
            self.assertEqual(extract_query(response, revised=revised), 'SELECT 2;')

    def test_correction_prefers_revised_answer(self):
        response = 'SQL: SELECT 1;\nRevised_SQL: SELECT 2;'
        self.assertEqual(extract_query(response, revised=True), 'SELECT 2;')
        self.assertEqual(extract_query('SQL: SELECT 1;', revised=True), 'SELECT 1;')

    def test_preserves_query_semantics(self):
        original = """WITH t AS (
            SELECT 'a;  b--c' AS x, 'line
break' AS y -- comment
        )
        SELECT x, y, 1/* comment */+2 AS [a;b] FROM t;
        """
        extracted = extract_query('```sql\n' + original + '\n```')
        with sqlite3.connect(':memory:') as db:
            self.assertEqual(db.execute(extracted).fetchall(), db.execute(original).fetchall())

    def test_quoted_escapes(self):
        query = '''SELECT 'it''s; fine', 2 AS "a"";b", 3 AS `a;b`;'''
        self.assertEqual(extract_query(query + '\nExplanation'), query)

    def test_reasoning_and_invalid_output(self):
        self.assertEqual(
            extract_query('<think>SQL: SELECT 0;</think>\nSELECT 1;'),
            'SELECT 1;',
        )
        for response in ('', 'No SQL generated', 'Use SELECT to answer.',
                         '<think>SELECT 1;', '```sql\nSELECT 1;'):
            with self.subTest(response=response):
                self.assertIsNone(extract_query(response))


if __name__ == '__main__':
    unittest.main()
