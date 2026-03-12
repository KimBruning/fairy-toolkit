#!/usr/bin/env python3
"""
Claude Artifact Extractor

Extracts artifact content from Claude conversation JSON exports.
Artifacts are stored in tool_use blocks where name == "artifacts".

Supports searching by keyword, listing all artifacts, and extracting
to individual files.

Usage:
    ./extract-artifacts.py FILE                        # List all artifacts
    ./extract-artifacts.py FILE --search "truth oath"  # Search artifact content
    ./extract-artifacts.py FILE --extract              # Extract all to files
    ./extract-artifacts.py FILE --extract -s "geas"    # Extract matching only
    ./extract-artifacts.py FILE --full                 # Print full content of all
    ./extract-artifacts.py FILE --full -s "police"     # Print full content of matches
    ./extract-artifacts.py DIRECTORY -s "Gwyneth"      # Search across all conversations
"""

import argparse
import json
import re
import sys
from pathlib import Path


def sanitize_filename(name: str) -> str:
    """Convert artifact title to safe filename."""
    safe = re.sub(r'[<>:"/\\|?*]', '_', name)
    safe = re.sub(r'\s+', '_', safe)
    safe = safe.strip('._')
    if len(safe) > 100:
        safe = safe[:100]
    return safe or "unnamed"


def find_artifacts(data: dict) -> list[dict]:
    """Recursively find all artifact tool_use blocks in a conversation."""
    artifacts = []

    def walk(obj):
        if isinstance(obj, dict):
            if obj.get('type') == 'tool_use' and obj.get('name') == 'artifacts':
                inp = obj.get('input', {})
                artifacts.append({
                    'id': inp.get('id', ''),
                    'type': inp.get('type', ''),
                    'title': inp.get('title', ''),
                    'command': inp.get('command', ''),
                    'content': inp.get('content', ''),
                })
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(data)
    return artifacts


def deduplicate_artifacts(artifacts: list[dict]) -> list[dict]:
    """Keep latest version of each artifact by id, merge updates."""
    by_id = {}
    for a in artifacts:
        aid = a['id']
        if not aid:
            # No id - treat as unique
            by_id[id(a)] = a
            continue

        if aid not in by_id:
            by_id[aid] = {**a, 'versions': 1}
        else:
            by_id[aid]['versions'] += 1
            # Updates may have content or title
            if a['content']:
                by_id[aid]['content'] = a['content']
            if a['title']:
                by_id[aid]['title'] = a['title']

    return list(by_id.values())


def matches_search(artifact: dict, terms: list[str]) -> bool:
    """Check if artifact matches all search terms (case-insensitive)."""
    text = (artifact.get('title', '') + ' ' + artifact.get('content', '')).lower()
    return all(t.lower() in text for t in terms)


def process_file(path: Path, search_terms: list[str] | None = None,
                 dedupe: bool = True) -> list[dict]:
    """Load a conversation file and return its artifacts."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return []

    conv_name = data.get('name', path.stem)
    artifacts = find_artifacts(data)

    if dedupe:
        artifacts = deduplicate_artifacts(artifacts)

    # Tag with source
    for a in artifacts:
        a['source_file'] = path.name
        a['source_conversation'] = conv_name

    if search_terms:
        artifacts = [a for a in artifacts if matches_search(a, search_terms)]

    return artifacts


def print_list(artifacts: list[dict], show_source: bool = False):
    """Print a summary table of artifacts."""
    if not artifacts:
        print("No artifacts found.")
        return

    print(f"{'#':>3}  {'Title':<55} {'Type':<12} {'Size':>6}")
    print('-' * 80)

    for i, a in enumerate(artifacts):
        title = a.get('title', '(untitled)') or '(untitled)'
        if len(title) > 53:
            title = title[:50] + '...'
        atype = a.get('type', '?')
        size = len(a.get('content', ''))
        versions = a.get('versions', 1)
        vstr = f" v{versions}" if versions > 1 else ""
        source = f"  [{a['source_file'][:40]}]" if show_source else ""
        print(f"{i:>3}  {title:<55} {atype:<12} {size:>5}b{vstr}{source}")

    print(f"\n{len(artifacts)} artifacts")


def print_full(artifacts: list[dict]):
    """Print full content of all artifacts."""
    if not artifacts:
        print("No artifacts found.")
        return

    for i, a in enumerate(artifacts):
        title = a.get('title', '(untitled)') or '(untitled)'
        content = a.get('content', '')
        source = a.get('source_conversation', '')

        print(f"{'=' * 72}")
        print(f"ARTIFACT {i}: {title}")
        if source:
            print(f"Source: {source}")
        print(f"Type: {a.get('type', '?')}  |  Size: {len(content)} chars")
        print(f"{'=' * 72}")
        if content:
            print(content)
        else:
            print("(no content)")
        print()


def extract_to_files(artifacts: list[dict], output_dir: Path):
    """Write each artifact to its own file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for i, a in enumerate(artifacts):
        title = a.get('title', '') or f"artifact_{i}"
        content = a.get('content', '')
        if not content.strip():
            continue

        safe_name = sanitize_filename(title)
        atype = a.get('type', 'text')
        ext = '.md' if atype in ('text/markdown', 'text', '') else '.txt'

        output_file = output_dir / f"{safe_name}{ext}"
        counter = 1
        while output_file.exists():
            output_file = output_dir / f"{safe_name}_{counter}{ext}"
            counter += 1

        with open(output_file, 'w') as f:
            f.write(f"---\n")
            f.write(f"title: {title}\n")
            f.write(f"source_conversation: {a.get('source_conversation', '')}\n")
            f.write(f"source_file: {a.get('source_file', '')}\n")
            f.write(f"artifact_type: {atype}\n")
            versions = a.get('versions', 1)
            if versions > 1:
                f.write(f"versions: {versions}\n")
            f.write(f"---\n\n")
            f.write(content)

        written += 1

    print(f"Extracted {written} artifacts to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract artifacts from Claude conversation exports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s conversation.json                     List all artifacts
  %(prog)s conversation.json --full              Print full content
  %(prog)s conversation.json -s "truth oath"     Search for matching artifacts
  %(prog)s conversation.json --full -s "geas"    Full content of matches
  %(prog)s conversation.json --extract           Extract all to files
  %(prog)s ./dump_dir/ -s "Gwyneth"              Search across all conversations
""")
    parser.add_argument('input', type=Path,
                        help='Conversation JSON file, or directory of them')
    parser.add_argument('-s', '--search', nargs='+', metavar='TERM',
                        help='Filter artifacts matching all search terms')
    parser.add_argument('--full', action='store_true',
                        help='Print full artifact content')
    parser.add_argument('--extract', action='store_true',
                        help='Extract artifacts to individual files')
    parser.add_argument('-o', '--output', type=Path, default=Path('./extracted_artifacts'),
                        help='Output directory for --extract (default: ./extracted_artifacts)')
    parser.add_argument('--no-dedupe', action='store_true',
                        help='Keep all versions instead of deduplicating by id')
    parser.add_argument('--json', action='store_true',
                        help='Output as JSON')

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    # Collect files to process
    if args.input.is_dir():
        files = sorted(args.input.glob('*.json'))
        if not files:
            print(f"Error: no JSON files found in {args.input}", file=sys.stderr)
            sys.exit(1)
        show_source = True
    else:
        files = [args.input]
        show_source = False

    # Process
    all_artifacts = []
    for path in files:
        try:
            artifacts = process_file(path, args.search, dedupe=not args.no_dedupe)
            all_artifacts.extend(artifacts)
        except Exception as e:
            print(f"Warning: {path.name}: {e}", file=sys.stderr)

    if not all_artifacts:
        print("No matching artifacts found.")
        sys.exit(0)

    # Output
    if args.json:
        print(json.dumps(all_artifacts, indent=2))
    elif args.extract:
        extract_to_files(all_artifacts, args.output)
    elif args.full:
        print_full(all_artifacts)
    else:
        print_list(all_artifacts, show_source=show_source)


if __name__ == '__main__':
    main()
