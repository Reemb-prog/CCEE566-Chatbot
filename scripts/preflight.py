"""Standard-library-only integrity and schema check; safe before installing dependencies."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'inputs' / 'asset_manifest.json'


def main() -> int:
    assert MANIFEST.exists(), 'Missing asset manifest'
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    errors = []
    print('RHU Claude chatbot harness: input audit')
    for item in manifest['assets']:
        path = ROOT / item['path']
        if not path.is_file():
            errors.append(f"MISSING: {item['path']}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item['sha256']:
            errors.append(f"HASH MISMATCH: {item['path']}")
        else:
            print(f"  OK   {item['path']} ({path.stat().st_size:,} bytes)")
    columns = {
        'RETRIEVAL_PASSAGES.csv': {'passage_id', 'passage_text', 'source_page_start', 'source_url'},
        'RHU_COURSE_DESCRIPTIONS.csv': {'course_code', 'credits', 'prerequisite_text', 'source_page'},
        'RHU_PROGRAM_STUDY_PLAN_ROWS.csv': {'program_or_section','semester','course_code','source_page'},
        'RHU_CATALOG_FULL_TEXT_BY_PAGE.csv': {'catalog_page','page_text'},
    }
    for fname, expected in columns.items():
        path = ROOT / 'inputs' / 'knowledge' / fname
        if not path.exists():
            continue
        with path.open(encoding='utf-8-sig', newline='') as f:
            rdr = csv.DictReader(f)
            actual = set(rdr.fieldnames or [])
            if not expected.issubset(actual):
                errors.append(f'{fname} missing columns: {sorted(expected - actual)}')
    if errors:
        for error in errors:
            print('ERROR:', error)
        return 1
    print(f"PASS: {len(manifest['assets'])} source assets checked; CSV schemas verified")
    print('Next: run Claude Code using RUN_CLAUDE.bat or RUN_CLAUDE.sh.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
