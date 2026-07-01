# Workflow: Create a Preference Dataset

1. Create a preference project.
2. Write the prompt.
3. Write the chosen response.
4. Write the rejected response.
5. Explain why chosen is better.
6. Validate the pair.
7. Use Preference Review to inspect chosen/rejected contrast.
8. Optionally prepare an AI Assist preference-strength judge pass.
9. Export preference JSONL or a preference ranking artifact.

## Example row

```json
{"prompt":"Explain recursion simply.","chosen":"Recursion is when a function calls itself.","rejected":"Recursion is a programming thing where stuff happens again."}
```
