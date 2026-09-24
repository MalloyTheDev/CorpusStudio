"""The full-parameter SFT worker's torch-free surface: the fixed-length row padding (the training loop is
# pragma, proven by a run - mirrors the pretraining worker's base-gate coverage). The sealed truncation
policy and the full-content preflight are covered in ``test_full_finetune_preflight.py``."""

import pytest

from corpus_studio.training.full_finetune_trainer import FullFinetuneDataRefusal, pad_sft_row


def test_pad_sft_row_pads_and_masks_to_seq_len():
    out = pad_sft_row([5, 6, 7], seq_len=6, pad_id=0)
    assert out["input_ids"] == [5, 6, 7, 0, 0, 0]
    assert out["labels"] == [5, 6, 7, -100, -100, -100]  # never train on padding
    assert out["attention_mask"] == [1, 1, 1, 0, 0, 0]


def test_pad_sft_row_refuses_overlength_instead_of_slicing():
    # Truncation is a sealed-policy decision made (and counted) before padding; the row builder never
    # hides it by slicing (#861).
    with pytest.raises(FullFinetuneDataRefusal, match="5 tokens, beyond max_sequence_len=3") as refused:
        pad_sft_row([1, 2, 3, 4, 5], seq_len=3, pad_id=0)
    assert str(refused.value).isascii()


def test_pad_sft_row_refuses_an_empty_row_instead_of_training_padding():
    with pytest.raises(FullFinetuneDataRefusal, match="zero tokens"):
        pad_sft_row([], seq_len=3, pad_id=0)


def test_pad_sft_row_exact_length_has_no_padding():
    out = pad_sft_row([9, 8], seq_len=2, pad_id=7)
    assert out["input_ids"] == [9, 8]
    assert out["labels"] == [9, 8]
    assert out["attention_mask"] == [1, 1]
