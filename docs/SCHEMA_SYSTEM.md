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
  "fields": [
    {"name": "instruction", "type": "text", "required": true},
    {"name": "input", "type": "text", "required": false},
    {"name": "output", "type": "markdown", "required": true},
    {"name": "tags", "type": "list", "required": false}
  ]
}
```

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
