# Schema Examples

Corpus Studio ships eight built-in schemas. Each carries a `description` and a valid `example` row; the desktop new-project dialog shows both and pre-fills the editor with the example so the correct format is obvious. Each line below is a single JSONL row.

The current validator checks that each row is a JSON object, required fields are present and non-empty, declared field types match, and chat messages include valid role/content structure.

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

## Code

Required fields: `task`, `language`, `instruction`, `output`

```json
{"task":"function-implementation","language":"python","instruction":"Write a function that returns the factorial of a non-negative integer.","input":"","output":"def factorial(n): return 1 if n < 2 else n * factorial(n - 1)","tests":["assert factorial(0) == 1","assert factorial(5) == 120"]}
```

Use code rows for code-generation fine-tuning. `tests` can hold assertions used to check generated output.

## Image caption

Required fields: `image`, `caption`

```json
{"image":"images/golden_retriever.jpg","caption":"A golden retriever sitting on a grass lawn in bright sunlight.","tags":["animal","dog"],"license":"CC-BY-4.0"}
```

Use image-caption rows for vision-language datasets. `image` is a file path relative to the project.

## Retrieval

Required fields: `query`, `positive`

```json
{"query":"How do I reverse a list in Python?","positive":"Use slicing with a step of -1: reversed_list = original[::-1].","negative":"Python lists are ordered, mutable collections of items.","source":"python_docs"}
```

Use retrieval rows for embedding or reranker training, pairing a query with a relevant passage and an optional hard negative.

## Evaluation

Required fields: `id`, `prompt`, `expected_answer`

```json
{"id":"eval-001","prompt":"What is the time complexity of binary search?","expected_answer":"O(log n).","rubric":"Full credit for O(log n); partial credit for mentioning logarithmic time.","category":"algorithms"}
```

Use evaluation rows for held-out Evaluation Lab test sets. Keep these separate from training data.

## Validate a row file

Save rows to a `.jsonl` file and run:

```powershell
.\engine\.venv\Scripts\python.exe -m corpus_studio.cli validate path\to\file.jsonl instruction
```

Replace `instruction` with `raw_text`, `chat`, or `preference` for the other schemas.
