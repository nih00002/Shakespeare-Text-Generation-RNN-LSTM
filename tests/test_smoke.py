"""Small CPU tests. No training epochs are run."""
import unittest
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

from rnn import RNN
from sample import sample
from create_sequence import create_sequence


class RecurrentSmokeTests(unittest.TestCase):
    def test_forward_and_backward_for_both_architectures(self):
        torch.manual_seed(0)
        for kind in ("basic_rnn", "lstm_rnn"):
            with self.subTest(kind=kind):
                model = RNN(vocab_size=7, hidden_size=5, num_layers=2, rnn_type=kind)
                tokens = torch.randint(0, 7, (2, 4))
                x = F.one_hot(tokens, num_classes=7).float()
                h, c = model.initial_hidden_states(batch_size=2)
                logits, final_h, final_c = model(x, h, c)
                self.assertEqual(tuple(logits.shape), (8, 7))
                self.assertEqual(tuple(final_h.shape), (2, 2, 5))
                self.assertEqual(tuple(final_c.shape), (2, 2, 5))
                logits.square().mean().backward()
                self.assertIsNotNone(model.rnn.rnn[0].W.grad)

    def test_sampling_keeps_seed_without_duplicating_warmup(self):
        torch.manual_seed(0)
        alphabet = "abcde"
        model = RNN(vocab_size=len(alphabet), hidden_size=5, rnn_type="lstm_rnn")
        char_to_index = {c: i for i, c in enumerate(alphabet)}
        index_to_char = dict(enumerate(alphabet))
        result = sample(model, "ab", 12, char_to_index, index_to_char, top_k=3)
        self.assertTrue(result.startswith("ab"))
        self.assertEqual(len(result), 14)
        self.assertTrue(set(result).issubset(set(alphabet)))

    def test_sequence_preparation(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "tiny.txt"
            p.write_text("abcdefghijklmnopqrstuvwxyz" * 6)
            training, validation, lookup, reverse = create_sequence(str(p), seq_len=5)
            self.assertGreater(len(training), 0)
            self.assertGreater(len(validation), 0)
            self.assertEqual(len(lookup), len(reverse))
            self.assertEqual(tuple(training.tensors[0].shape[1:]), (5,))


if __name__ == "__main__":
    unittest.main()
