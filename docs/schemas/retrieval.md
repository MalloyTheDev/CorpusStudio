# Retrieval Schema

Used for embedding, reranker, semantic search, and RAG evaluation datasets.

## Example

```json
{
  "query": "How do I reset my password?",
  "positive": "To reset your password, open Account Settings.",
  "negative": "Our refund policy allows returns within 30 days."
}
```

## Fields

- query
- positive
- negative
- hard_negative
- relevance_score
- source
