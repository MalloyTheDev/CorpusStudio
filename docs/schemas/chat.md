# Chat Schema

Used for multi-turn conversational tuning.

## Minimal JSONL

```json
{"messages":[{"role":"user","content":"What is recursion?"},{"role":"assistant","content":"Recursion is when a function calls itself."}]}
```

## Required fields

- messages

Each message requires:

- role
- content

Valid roles:

- system
- user
- assistant
- tool
