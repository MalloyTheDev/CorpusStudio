# Built-in Schemas

Field-level reference for each built-in schema (see `schemas/builtin/`). One
section per schema, consolidated from the former per-schema files. For the schema
*system* (how schemas are defined, validated, and extended) see
[`../SCHEMA_SYSTEM.md`](../SCHEMA_SYSTEM.md).


---

## Raw Text Schema

Used for pretraining corpora and continued pretraining.

### Minimal JSONL

```json
{"text":"A compiler translates source code into a lower-level representation."}
```

### Recommended fields

- text
- source
- domain
- license
- language
- token_count
- quality_score


---

## Instruction Schema

Used for instruction tuning.

### Minimal JSONL

```json
{"instruction":"Explain variables.","input":"","output":"A variable stores a value."}
```

### Required fields

- instruction
- output

### Optional fields

- input
- tags
- difficulty
- domain
- source
- license


---

## Chat Schema

Used for multi-turn conversational tuning.

### Minimal JSONL

```json
{"messages":[{"role":"user","content":"What is recursion?"},{"role":"assistant","content":"Recursion is when a function calls itself."}]}
```

### Required fields

- messages

Each message requires:

- role
- content

Valid roles:

- system
- user
- assistant
- tool


---

## Preference Schema

Used for DPO, ORPO, reward modeling, and ranking datasets.

### Minimal JSONL

```json
{"prompt":"Explain recursion.","chosen":"A clear answer.","rejected":"A weak answer."}
```

### Required fields

- prompt
- chosen
- rejected

### Optional fields

- reason
- quality_dimensions
- tags


---

## Code Schema

Used for code model training.

### Example

```json
{
  "task": "bug_fix",
  "language": "python",
  "instruction": "Fix the bug.",
  "input": "def divide(a, b):\n    return a / b",
  "output": "def divide(a, b):\n    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b",
  "tests": ["assert divide(10, 2) == 5"]
}
```

### Task types

- code_completion
- bug_fix
- code_explanation
- test_generation
- refactor
- documentation
- function_implementation
- security_review


---

## Image-Caption Schema

Used for image captioning, image generation, vision-language, and pixel-art datasets.

### Example

```json
{
  "image": "images/fire_mage_idle_01.png",
  "caption": "64x64 pixel art fire mage holding a small flame, transparent background",
  "tags": ["pixel-art", "mage", "idle"]
}
```

### Fields

- image
- caption
- negative_caption
- tags
- style
- resolution
- source
- license


---

## Retrieval Schema

Used for embedding, reranker, semantic search, and RAG evaluation datasets.

### Example

```json
{
  "query": "How do I reset my password?",
  "positive": "To reset your password, open Account Settings.",
  "negative": "Our refund policy allows returns within 30 days."
}
```

### Fields

- query
- positive
- negative
- hard_negative
- relevance_score
- source


---

## Evaluation Schema

Used for model evaluation and regression testing.

### Example

```json
{
  "id": "math_001",
  "prompt": "What is 12 * 8?",
  "expected_answer": "96",
  "rubric": "Answer must be exactly 96.",
  "category": "arithmetic"
}
```

### Fields

- id
- prompt
- expected_answer
- rubric
- category
- difficulty
- tags
