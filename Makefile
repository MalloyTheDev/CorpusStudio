.PHONY: test-engine validate-examples

test-engine:
	cd engine && PYTHONPATH=. pytest -q

validate-examples:
	cd engine && PYTHONPATH=. python -m corpus_studio.cli validate ../examples/datasets/instruction/train.jsonl instruction
	cd engine && PYTHONPATH=. python -m corpus_studio.cli validate ../examples/datasets/chat/train.jsonl chat
	cd engine && PYTHONPATH=. python -m corpus_studio.cli validate ../examples/datasets/preference/train.jsonl preference
