# Schema Examples

Corpus Studio v0.1 focuses on four authoring schemas in the desktop app: raw text, instruction, chat, and preference. Each line below is a single JSONL row.

The current validator checks that each row is valid JSON and that required fields are present and non-empty. Type-specific validation will become stricter as the schema engine matures.

## Raw text

Required field: `text`

```json
{"text":"A compiler translates source code into a lower-level representation.","source":"developer_notes.md","domain":"programming","license":"MIT","language":"en"}
```

Use raw text rows for pretraining or continued-pretraining corpora where the example is mainly source text plus provenance metadata.

## Instruction

Required fields: `instruction`, `output`

```json
{"instruction":"Explain what a variable is.","input":"","output":"A variable stores a value so a program can reuse or change it later.","tags":["programming","beginner"]}
```

Use instruction rows for supervised fine-tuning where the model should learn to answer a direct task.

## Chat

Required field: `messages`

```json
{"messages":[{"role":"system","content":"Answer like a concise programming tutor."},{"role":"user","content":"What is recursion?"},{"role":"assistant","content":"Recursion is when a function solves a problem by calling itself on a smaller version of that problem."}],"tags":["programming","concepts"]}
```

Use chat rows for multi-turn conversational tuning. Supported message roles are `system`, `user`, `assistant`, and `tool`.

## Preference

Required fields: `prompt`, `chosen`, `rejected`

```json
{"prompt":"Explain recursion in one sentence.","chosen":"Recursion is when a function calls itself to solve smaller pieces of a problem.","rejected":"Recursion is when code repeats somehow.","reason":"The chosen response is specific and technically correct."}
```

Use preference rows for DPO, ORPO, reward modeling, or ranking datasets where one answer should be preferred over another.

## Validate a row file

Save rows to a `.jsonl` file and run:

```powershell
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli validate path\to\file.jsonl instruction
```

Replace `instruction` with `raw_text`, `chat`, or `preference` for the other schemas.
