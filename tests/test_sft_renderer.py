from __future__ import annotations

import unittest

from main import _render_sft_row
from main import _assert_no_cross_sequence_lowpass_segments, _pad_sft_row_to_block_multiple


def char_tokenize(text: str) -> list[int]:
    return [ord(ch) for ch in text]


class SftRendererTests(unittest.TestCase):
    def test_renders_messages_schema(self) -> None:
        row = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }

        rendered = _render_sft_row(row, char_tokenize, seq_len=1024)

        self.assertIsNotNone(rendered)
        input_ids, labels = rendered
        self.assertEqual(len(input_ids), len(labels))
        supervised = [label for label in labels if label != -100]
        self.assertEqual(supervised, char_tokenize("Hi<|im_end|>\n"))

    def test_still_renders_sharegpt_schema(self) -> None:
        row = {
            "conversations": [
                {"from": "human", "value": "Hello"},
                {"from": "gpt", "value": "Hi"},
            ]
        }

        rendered = _render_sft_row(row, char_tokenize, seq_len=1024)

        self.assertIsNotNone(rendered)
        input_ids, labels = rendered
        self.assertEqual(len(input_ids), len(labels))
        supervised = [label for label in labels if label != -100]
        self.assertEqual(supervised, char_tokenize("Hi<|im_end|>\n"))

    def test_pads_sft_row_to_configured_block_multiple(self) -> None:
        input_ids = [1, 2, 3, 4, 5]
        labels = [-100, -100, 3, 4, 5]

        padded_ids, padded_labels, attention_mask, pad_count = _pad_sft_row_to_block_multiple(
            input_ids,
            labels,
            block_size=4,
            pad_token_id=0,
            seq_len=16,
        )

        self.assertEqual(len(padded_ids), 8)
        self.assertEqual(pad_count, 3)
        self.assertEqual(padded_ids[-3:], [0, 0, 0])
        self.assertEqual(padded_labels[-3:], [-100, -100, -100])
        self.assertEqual(attention_mask, [1, 1, 1, 1, 1, 0, 0, 0])

    def test_lowpass_segment_assertion_allows_padding_boundary(self) -> None:
        segments = [0, 0, -1, -1, 1, 1, 1, -1]

        _assert_no_cross_sequence_lowpass_segments(segments, block_size=4)

    def test_lowpass_segment_assertion_rejects_cross_conversation_block(self) -> None:
        segments = [0, 0, 1, -1]

        with self.assertRaises(RuntimeError):
            _assert_no_cross_sequence_lowpass_segments(segments, block_size=4)


if __name__ == "__main__":
    unittest.main()
