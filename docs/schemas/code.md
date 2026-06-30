# Code Schema

Used for code model training.

## Example

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

## Task types

- code_completion
- bug_fix
- code_explanation
- test_generation
- refactor
- documentation
- function_implementation
- security_review
