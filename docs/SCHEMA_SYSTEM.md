# Schema System

Corpus Studio is schema-driven.

A schema defines:

- dataset type
- fields
- required fields
- field types
- validation rules
- editor hints
- export mappings
- quality checks
- evaluation prompt and expected-output hints
- training format compatibility metadata

## Why schema-driven?

A hardcoded app becomes brittle:

```text
Instruction editor
Chat editor
Preference editor
Code editor
Image editor
...
```

A schema-driven app can use the same underlying engine for many dataset types.

## Built-in schema example

```json
{
  "id": "instruction",
  "name": "Instruction Dataset",
  "version": "0.1.0",
  "description": "Single-turn supervised fine-tuning: an instruction (with optional input) paired with the target output.",
  "fields": [
    {"name": "instruction", "type": "text", "required": true},
    {"name": "input", "type": "text", "required": false},
    {"name": "output", "type": "markdown", "required": true},
    {"name": "tags", "type": "list", "required": false}
  ],
  "example": {
    "instruction": "Explain what a variable is.",
    "input": "",
    "output": "A variable stores a value so a program can reuse or change it later.",
    "tags": ["programming", "beginner"]
  }
}
```

Each built-in schema carries a `description` and a valid `example` row. The
desktop new-project dialog shows both, and pre-fills the editor with the example
so the correct format is obvious. A test (`engine/tests/test_schema_examples.py`)
asserts every example validates against its own schema.

## Field types

Initial field types:

- string
- text
- markdown
- integer
- float
- boolean
- list
- object
- messages
- file_path
- image_path
- code

## Validation levels

1. structural validation
2. semantic validation
3. quality validation
4. export compatibility validation
5. evaluation readiness validation
6. training compatibility validation

## Current structural checks

The engine currently rejects:

- non-object JSONL rows
- missing required fields
- empty required strings, lists, and objects
- values that do not match the declared field type
- chat messages without a valid `system`, `user`, `assistant`, or `tool` role
- chat messages with missing or empty string content

## Example lifecycle

```text
draft -> valid -> reviewed -> split -> evaluated -> exported
```

Training preparation is staged after export:

```text
exported -> training config generated -> local training run
```

## Lab schema hints

Evaluation Lab, AI Assist Lab, and Training Lab features should read schema
metadata instead of hardcoding dataset behavior in the UI.

Useful hints include:

- which field becomes the model prompt
- which field is the expected output
- which field stores chat messages
- which fields are safe for tags
- which fields should be hidden from the model during evaluation
- which export formats are compatible with training tools

Evaluation examples must remain separate from training examples so test results
measure generalization rather than memorization.


---

## Copyable row examples

_Consolidated from the former SCHEMA_EXAMPLES.md._

### Schema Examples

Corpus Studio ships nine built-in schemas. Each carries a `description` and a valid `example` row; the desktop new-project dialog shows both and pre-fills the editor with the example so the correct format is obvious. Each line below is a single JSONL row.

The current validator checks that each row is a JSON object, required fields are present and non-empty, declared field types match, and chat messages include valid role/content structure.

#### Raw text

Required field: `text`

```json
{"text":"A compiler translates source code into a lower-level representation.","source":"developer_notes.md","domain":"programming","license":"MIT","language":"en"}
```

Use raw text rows for pretraining or continued-pretraining corpora where the example is mainly source text plus provenance metadata.

#### Instruction

Required fields: `instruction`, `output`

```json
{"instruction":"Explain what a variable is.","input":"","output":"A variable stores a value so a program can reuse or change it later.","tags":["programming","beginner"]}
```

Use instruction rows for supervised fine-tuning where the model should learn to answer a direct task.

#### Chat

Required field: `messages`

```json
{"messages":[{"role":"system","content":"Answer like a concise programming tutor."},{"role":"user","content":"What is recursion?"},{"role":"assistant","content":"Recursion is when a function solves a problem by calling itself on a smaller version of that problem."}],"tags":["programming","concepts"]}
```

Use chat rows for multi-turn conversational tuning. Supported message roles are `system`, `user`, `assistant`, and `tool`.

#### Preference

Required fields: `prompt`, `chosen`, `rejected`

```json
{"prompt":"Explain recursion in one sentence.","chosen":"Recursion is when a function calls itself to solve smaller pieces of a problem.","rejected":"Recursion is when code repeats somehow.","reason":"The chosen response is specific and technically correct."}
```

Use preference rows for DPO, ORPO, reward modeling, or ranking datasets where one answer should be preferred over another.

#### Code

Required fields: `task`, `language`, `instruction`, `output`

```json
{"task":"function-implementation","language":"python","instruction":"Write a function that returns the factorial of a non-negative integer.","input":"","output":"def factorial(n): return 1 if n < 2 else n * factorial(n - 1)","tests":["assert factorial(0) == 1","assert factorial(5) == 120"]}
```

Use code rows for code-generation fine-tuning. `tests` can hold assertions used to check generated output.

#### Image caption

Required fields: `image`, `caption`

```json
{"image":"images/golden_retriever.jpg","caption":"A golden retriever sitting on a grass lawn in bright sunlight.","tags":["animal","dog"],"license":"CC-BY-4.0"}
```

Use image-caption rows for vision-language datasets. `image` is a file path relative to the project.

#### Classification

Required fields: `text`, `label`

```json
{"text":"The delivery was fast and the product works exactly as described.","label":"positive","tags":["sentiment","product-review"]}
```

Use classification rows for text classification (sentiment, topic, intent, …). The label set is project-defined; pin `label` to a fixed `enum` in a project copy of the schema if you want validation to reject unknown labels.

#### Retrieval

Required fields: `query`, `positive`

```json
{"query":"How do I reverse a list in Python?","positive":"Use slicing with a step of -1: reversed_list = original[::-1].","negative":"Python lists are ordered, mutable collections of items.","source":"python_docs"}
```

Use retrieval rows for embedding or reranker training, pairing a query with a relevant passage and an optional hard negative.

#### Evaluation

Required fields: `id`, `prompt`, `expected_answer`

```json
{"id":"eval-001","prompt":"What is the time complexity of binary search?","expected_answer":"O(log n).","rubric":"Full credit for O(log n); partial credit for mentioning logarithmic time.","category":"algorithms"}
```

Use evaluation rows for held-out Evaluation Lab test sets. Keep these separate from training data.

#### Validate a row file

Save rows to a `.jsonl` file and run:

```powershell
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli validate path\to\file.jsonl instruction
```

Replace `instruction` with `raw_text`, `chat`, or `preference` for the other schemas.
