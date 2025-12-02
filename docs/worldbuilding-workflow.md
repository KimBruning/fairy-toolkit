# Worldbuilding Extraction Workflow

Using Claude Code with the toolkit to extract and maintain worldbuilding from stories.

## Overview

As you write stories, details accumulate: characters, locations, rules of magic, political structures. This workflow uses Claude Code to extract those details into structured worldbuilding docs that stay searchable alongside your stories.

## Setup

1. Run Claude Code from the parent directory (containing both toolkit and content)
2. Ensure content is indexed: `./fairy-toolkit/index.py`

## Extracting Characters and Locations

### 1. Search for stories to process

```bash
./fairy-toolkit/search.py "character name or theme"
```

### 2. Ask Claude to read and extract

Example prompt:
> Read the honeycake heist and the broom, and extract character sheets for each, and locations

Claude will:
- Use the search tool to find the stories
- Read the full text
- Identify characters with their traits, relations, species, roles
- Identify locations with type, features, rulers

### 3. Review the extraction

Claude presents the extracted info in a structured format. Review for:
- Accuracy (does it match what's in the story?)
- Completeness (any missing characters/locations?)
- Consistency (conflicts with existing worldbuilding?)

### 4. Write to worldbuilding files

Approve Claude's suggestion to write files. Structure:

```
worldbuilding/
├── characters/
│   ├── gwyneth.md
│   ├── caoimhe.md
│   └── ...
└── locations/
    ├── underhill.md
    ├── silverton-academy.md
    └── ...
```

One file per entity makes updates easier and searching more precise.

### 5. Re-index

```bash
./fairy-toolkit/index.py
```

Now the new worldbuilding is searchable alongside stories.

## Tips

- **Incremental extraction**: Process a few stories at a time rather than everything at once
- **Verify against canon**: If Claude extracts something that contradicts established worldbuilding, decide which version is correct
- **Add context**: After extraction, you can ask Claude to flesh out sparse entries based on other stories mentioning that character/location
- **Cross-reference**: Use search to find all mentions of a character across stories before finalizing their sheet

## Example Session

```
You: read the honeycake heist and the broom, extract character sheets and locations

Claude: [uses search tool to find stories]
Claude: [reads both files]
Claude: [presents extracted characters and locations]

You: sounds like a start!

Claude: [creates worldbuilding/characters/*.md and worldbuilding/locations/*.md]

You: you can now re-index

Claude: [runs ./fairy-toolkit/index.py]
```
