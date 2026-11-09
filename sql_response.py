"""Extract SQLite queries from model responses without changing SQL literals."""

import re


# Quoted tokens come first so comments/semicolons inside them remain literal.
_TOKENS = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]"
    r"|--[^\r\n]*|/\*[\s\S]*?\*/|\s+|;|.",
    re.DOTALL,
)


def normalize_sql(sql: str) -> str:
    """Remove comments and flatten whitespace outside quoted values/identifiers.

    Stop at the first unquoted semicolon, excluding any following explanation.
    Newlines inside quoted values are preserved to avoid changing their meaning.
    """
    parts = []
    for match in _TOKENS.finditer(sql):
        token = match.group()
        if token.startswith(('--', '/*')) or token.isspace():
            parts.append(' ')
        else:
            parts.append(token)
        if token == ';':
            break
    return ''.join(parts).strip()


def extract_query(response: str, revised: bool = False):
    """Prefer the final SQL fence, then legacy labels, then a bare query.

    Do not search arbitrary prose for SELECT: it may describe an example rather
    than the answer. Closed reasoning blocks are excluded from consideration.
    """
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.S | re.I).strip()
    if re.search(r'<think>', response, re.I):
        return None
    labels = ('Revised_SQL', 'SQL') if revised else ('SQL',)
    fences = re.findall(r'```(?:sql|sqlite)?[ \t]*\r?\n(.*?)```', response, re.S | re.I)
    candidates = list(reversed(fences))
    for label in labels:
        matches = list(re.finditer(
            rf'^\s*(?:\*\*)?{label}\s*:(?:\*\*)?\s*', response, re.M | re.I
        ))
        candidates.extend(response[m.end():].strip() for m in reversed(matches))
    candidates.append(response)
    for candidate in candidates:
        fenced = re.match(r'```(?:sql|sqlite)?[ \t]*\r?\n(.*?)```', candidate, re.S | re.I)
        if fenced:
            candidate = fenced.group(1)
        # An unclosed fence is potentially truncated output.
        if '```' in candidate:
            continue
        query = normalize_sql(candidate)
        if re.match(r'^(?:SELECT|WITH)\b', query, re.I):
            return query
    return None
