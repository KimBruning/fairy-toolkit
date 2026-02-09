#!/usr/bin/env python3
"""
List stories with YAML frontmatter metadata.

Usage:
    ./list-stories.py                    # List all stories
    ./list-stories.py --sort fairy_age   # Sort by fairy age
    ./list-stories.py --sort era         # Sort by era
    ./list-stories.py --filter era=academy  # Filter by field
    ./list-stories.py --missing          # Show stories without frontmatter
"""

import argparse
import sys
from pathlib import Path
import yaml
import re

# Era ordering for sorting
ERA_ORDER = {
    'early-transition': 0,
    'toddler': 1,
    'young-child': 2,
    'pre-academy': 3,
    'academy': 4,
    'young-adult': 5,
    'unknown': 99,
}

def parse_frontmatter(path: Path) -> dict | None:
    """Extract YAML frontmatter from a file."""
    try:
        content = path.read_text(encoding='utf-8')
    except Exception:
        return None

    # Check for YAML frontmatter (--- at start)
    if not content.startswith('---'):
        return None

    # Find closing ---
    match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not match:
        return None

    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None


def parse_age(age_str) -> float:
    """Parse fairy_age field to sortable number."""
    if age_str is None:
        return 999
    if isinstance(age_str, (int, float)):
        return float(age_str)

    # Handle strings like "~0.5", "4-5", "~10"
    s = str(age_str).strip()
    s = s.lstrip('~')

    if '-' in s:
        # Range like "4-5" -> use midpoint
        parts = s.split('-')
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return 999

    try:
        return float(s)
    except ValueError:
        return 999


def main():
    parser = argparse.ArgumentParser(description='List stories with metadata')
    parser.add_argument('--root', type=Path, default=Path('../fairy_project/stories'),
                        help='Stories directory')
    parser.add_argument('--sort', choices=['name', 'fairy_age', 'era', 'modified'],
                        default='name', help='Sort order')
    parser.add_argument('--filter', type=str, help='Filter by field=value')
    parser.add_argument('--missing', action='store_true',
                        help='Show only stories without frontmatter')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"Error: {root} not found", file=sys.stderr)
        sys.exit(1)

    # Collect stories
    stories = []
    for ext in ['*.md', '*.txt']:
        for path in root.glob(ext):
            if path.name.startswith('.'):
                continue
            meta = parse_frontmatter(path)
            stories.append({
                'path': path,
                'name': path.stem,
                'has_frontmatter': meta is not None,
                'meta': meta or {},
            })

    # Filter
    if args.missing:
        stories = [s for s in stories if not s['has_frontmatter']]

    if args.filter:
        key, _, value = args.filter.partition('=')
        stories = [s for s in stories if str(s['meta'].get(key, '')).lower() == value.lower()]

    # Sort
    if args.sort == 'name':
        stories.sort(key=lambda s: s['name'].lower())
    elif args.sort == 'fairy_age':
        stories.sort(key=lambda s: parse_age(s['meta'].get('fairy_age')))
    elif args.sort == 'era':
        stories.sort(key=lambda s: ERA_ORDER.get(s['meta'].get('era', 'unknown'), 99))
    elif args.sort == 'modified':
        stories.sort(key=lambda s: s['path'].stat().st_mtime, reverse=True)

    # Output
    if args.json:
        import json
        out = [{
            'name': s['name'],
            'has_frontmatter': s['has_frontmatter'],
            **s['meta']
        } for s in stories]
        print(json.dumps(out, indent=2, default=str))
    else:
        # Table output
        print(f"{'Story':<40} {'Era':<18} {'Age':<8} {'Characters'}")
        print('-' * 90)
        for s in stories:
            meta = s['meta']
            era = meta.get('era', '-')
            age = meta.get('fairy_age', '-')
            chars = meta.get('characters', [])
            if isinstance(chars, list):
                chars = ', '.join(chars[:3])
                if len(meta.get('characters', [])) > 3:
                    chars += '...'
            name = s['name'][:38]
            print(f"{name:<40} {era:<18} {str(age):<8} {chars}")

        print(f"\nTotal: {len(stories)} stories")
        with_fm = sum(1 for s in stories if s['has_frontmatter'])
        print(f"With frontmatter: {with_fm}, Without: {len(stories) - with_fm}")


if __name__ == '__main__':
    main()
