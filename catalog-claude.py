#!/usr/bin/env python3
"""
Catalog Claude conversation exports.

Scans split conversation JSON files and extracts metadata
for triage: titles, dates, artifact info, character mentions, sizes.

Usage:
    ./catalog-claude.py                          # Full catalog
    ./catalog-claude.py --fairy                  # Only fairy-related
    ./catalog-claude.py --artifacts              # Only conversations with artifacts
    ./catalog-claude.py --sort artifacts         # Sort by artifact count
    ./catalog-claude.py --json                   # JSON output
"""

import argparse
import json
import re
import sys
from pathlib import Path

KNOWN_CHARACTERS = [
    "Gwyneth", "Caoimhe", "Caiomhe", "Eirwen", "Rhia", "Eilis", "Brigid",
    "Tinu", "Ilendir", "Marion", "Selenis", "Lily", "Grace", "Dana",
    "Elena", "Angela", "Varina", "Emery", "Mrs. Elkins", "Mrs. Potter",
    "Mrs. Greenthumb",
]

FAIRY_KEYWORDS = [
    "fairy", "fairie", "gwyneth", "sylvarum", "underhill", "caoimhe",
    "caiomhe", "eirwen", "glamour", "magelight", "north forest",
    "changeling", "fae", "pixie", "gnome", "tinu", "ilendir",
]


def scan_conversation(path: Path) -> dict:
    """Extract metadata from a conversation JSON file."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    name = data.get('name', '')
    uuid = data.get('uuid', '')
    created = data.get('created_at', '')[:10]
    updated = data.get('updated_at', '')[:10]
    messages = data.get('chat_messages', [])

    # Count messages and total text
    human_msgs = 0
    assistant_msgs = 0
    total_text_len = 0
    all_text = []

    # Artifact tracking
    artifacts = []
    artifact_creates = {}  # id -> latest version

    for msg in messages:
        sender = msg.get('sender', '')
        text = msg.get('text', '') or ''
        total_text_len += len(text)
        all_text.append(text)

        if sender == 'human':
            human_msgs += 1
        elif sender == 'assistant':
            assistant_msgs += 1

        # Scan content blocks for artifacts
        for block in (msg.get('content') or []):
            if block.get('type') == 'tool_use' and block.get('name') == 'artifacts':
                inp = block.get('input', {})
                art_id = inp.get('id', '?')
                art_type = inp.get('type', '?')
                art_title = inp.get('title', '')
                art_cmd = inp.get('command', '?')
                art_content = inp.get('content', '')

                if art_cmd == 'create' or art_id not in artifact_creates:
                    artifact_creates[art_id] = {
                        'id': art_id,
                        'type': art_type,
                        'title': art_title,
                        'command': art_cmd,
                        'content_len': len(art_content),
                        'versions': 1,
                    }
                else:
                    artifact_creates[art_id]['versions'] += 1
                    artifact_creates[art_id]['content_len'] = max(
                        artifact_creates[art_id]['content_len'], len(art_content)
                    )
                    if art_title:
                        artifact_creates[art_id]['title'] = art_title

    artifacts = list(artifact_creates.values())

    # Character detection (word boundary matching)
    combined_text = ' '.join(all_text)
    characters_found = [c for c in KNOWN_CHARACTERS
                        if re.search(r'\b' + re.escape(c) + r'\b', combined_text, re.IGNORECASE)]

    # Fairy relevance (word boundary matching)
    fairy_matches = []
    for kw in FAIRY_KEYWORDS:
        m = re.search(r'\b' + re.escape(kw) + r'\b', combined_text, re.IGNORECASE)
        if m:
            # Grab context around the match
            idx = m.start()
            start = max(0, idx - 40)
            end = min(len(combined_text), idx + 40)
            fairy_matches.append({
                'keyword': kw,
                'context': combined_text[start:end].replace('\n', ' '),
            })
    if not fairy_matches:
        for kw in FAIRY_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', name, re.IGNORECASE):
                fairy_matches.append({'keyword': kw, 'context': f'(in title: {name})'})

    return {
        'file': path.name,
        'uuid': uuid,
        'name': name,
        'created': created,
        'updated': updated,
        'human_msgs': human_msgs,
        'assistant_msgs': assistant_msgs,
        'total_msgs': human_msgs + assistant_msgs,
        'total_text_kb': round(total_text_len / 1024, 1),
        'artifacts': artifacts,
        'artifact_count': len(artifacts),
        'characters': characters_found,
        'is_fairy': bool(fairy_matches),
        'fairy_matches': fairy_matches,
    }


def print_table(entries, show_artifacts=False, show_why=False):
    """Print a human-readable table."""
    print(f"{'Date':<12} {'Messages':>8} {'Size':>8} {'Art':>4} {'Characters':<30} {'Name'}")
    print('-' * 120)
    for e in entries:
        chars = ', '.join(e['characters'][:4])
        if len(e['characters']) > 4:
            chars += '...'
        fairy_marker = '*' if e['is_fairy'] else ' '
        size = f"{e['total_text_kb']}kb"
        print(f"{e['created']:<12} {e['total_msgs']:>8} {size:>8} {e['artifact_count']:>4} {chars:<30} {fairy_marker}{e['name'][:55]}")

        if show_artifacts and e['artifacts']:
            for a in e['artifacts']:
                vstr = f" (v{a['versions']})" if a['versions'] > 1 else ""
                print(f"{'':>37} [{a['type']:<20}] {a['title']}{vstr}")
        if show_why and e.get('fairy_matches'):
            for fm in e['fairy_matches']:
                print(f"{'':>14} ? {fm['keyword']}: ...{fm['context']}...")

    print(f"\nTotal: {len(entries)} conversations")
    fairy_count = sum(1 for e in entries if e['is_fairy'])
    art_count = sum(1 for e in entries if e['artifact_count'] > 0)
    total_arts = sum(e['artifact_count'] for e in entries)
    print(f"Fairy-related: {fairy_count} | With artifacts: {art_count} | Total artifacts: {total_arts}")


def main():
    parser = argparse.ArgumentParser(description='Catalog Claude conversation exports')
    parser.add_argument('--root', type=Path, default=Path('../claude_2026-03-11_dump'),
                        help='Directory of split conversation JSONs')
    parser.add_argument('--fairy', action='store_true', help='Only fairy-related conversations')
    parser.add_argument('--artifacts', action='store_true', help='Only conversations with artifacts')
    parser.add_argument('--sort', choices=['date', 'messages', 'size', 'artifacts', 'name'],
                        default='date', help='Sort order')
    parser.add_argument('--show-artifacts', action='store_true', help='Show artifact details')
    parser.add_argument('--why', action='store_true', help='Show why conversations were flagged as fairy-related')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    parser.add_argument('--out', type=Path, help='Write JSON catalog to file')
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"Error: {root} not found", file=sys.stderr)
        sys.exit(1)

    # Scan all conversations
    entries = []
    json_files = sorted(root.glob('*.json'))
    for i, path in enumerate(json_files):
        if (i + 1) % 200 == 0:
            print(f"  Scanning {i+1}/{len(json_files)}...", file=sys.stderr)
        try:
            entries.append(scan_conversation(path))
        except Exception as e:
            print(f"Warning: {path.name}: {e}", file=sys.stderr)

    print(f"Scanned {len(entries)} conversations", file=sys.stderr)

    # Filter
    if args.fairy:
        entries = [e for e in entries if e['is_fairy']]
    if args.artifacts:
        entries = [e for e in entries if e['artifact_count'] > 0]

    # Sort
    sort_keys = {
        'date': lambda e: e['created'],
        'messages': lambda e: e['total_msgs'],
        'size': lambda e: e['total_text_kb'],
        'artifacts': lambda e: e['artifact_count'],
        'name': lambda e: e['name'].lower(),
    }
    reverse = args.sort in ('messages', 'size', 'artifacts')
    entries.sort(key=sort_keys[args.sort], reverse=reverse)

    # Output
    if args.json or args.out:
        output = json.dumps(entries, indent=2)
        if args.out:
            args.out.write_text(output)
            print(f"Wrote catalog to {args.out}", file=sys.stderr)
        else:
            print(output)
    else:
        print_table(entries, show_artifacts=args.show_artifacts, show_why=args.why)


if __name__ == '__main__':
    main()
