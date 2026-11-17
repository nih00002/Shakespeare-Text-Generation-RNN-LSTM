# Shakespeare Text Generation with RNN and LSTM

**Character-level language modeling with custom recurrent neural networks in PyTorch.**

This repository isolates **Project 3** from a larger deep learning coursework archive. It implements a vanilla recurrent neural network (tanh RNN) and an LSTM using custom recurrent cells, trains either model to predict the next character in a text sequence, and samples new text using an initial prompt.

> **Scope and accuracy:** The default `data/shakespeare.txt` file is a Shakespeare-themed **dramatic dialogue corpus**, not exclusively a collection of sonnets or a verified edition of one particular poem. A separate `data/shakespeare_sonnets.txt` file is supplied for experimentation. The included generated-text examples are outputs from the original project archive; **trained `.pt` checkpoints and full training logs were not included**. This repository does not claim a particular benchmark accuracy.

## Core ideas

- **Vanilla RNN:** manually implemented tanh recurrent cell, with an input matrix, recurrent matrix, and bias.
- **LSTM:** manually implemented input, forget, output, and candidate gates with separate cell state.
- **Character encoding:** each character is mapped to an index and represented as a one-hot vector.
- **Sequence modeling:** fixed-length character windows train next-character prediction.
- **Generation:** sample one character at a time using top-*k* sampling and a text prompt.
- **Validation:** the supplied preprocessing code makes a contiguous train/validation split of character sequences (85%/15%).

## Repository structure

```text
Shakespeare-Text-Generation-RNN-LSTM/
├── README.md
├── requirements.txt
├── .gitignore
├── basic_rnn_cell.py       # tanh recurrent cell
├── lstm_cell.py            # LSTM recurrent cell
├── create_rnn.py           # stacked recurrent layers
├── rnn.py                  # model and output classifier
├── create_sequence.py      # character encoding and data split
├── train.py                # optimization, validation and plots
├── sample.py               # text generation
├── text_generation.py      # command-line entry point
├── data/
│   ├── shakespeare.txt
│   └── shakespeare_sonnets.txt
├── examples/               # samples supplied in original archive
├── tests/                  # fast forward/backward and generation checks
├── archive/assignment_starters/  # incomplete templates, not entry points
└── docs/                   # original coursework disclosure
```

## Installation

Requires Python and PyTorch. For a local environment:

```bash
python -m venv .venv
# On macOS/Linux: source .venv/bin/activate
# On Windows:     .venv\Scripts\activate
pip install -r requirements.txt
```

> Installing PyTorch can vary by CPU/CUDA configuration. Refer to the official PyTorch installation instructions when using a GPU.

## Training and generation

Run commands from the **repository root**. The default training configuration in `text_generation.py` specifies **120 epochs, learning rate 0.01, batch size 32, weight decay 0.0001, and sequence length 80**. These are the configurations in the supplied project, **not measurements of a new training run**.

Vanilla RNN (two recurrent layers, 96 hidden units):

```bash
python text_generation.py --rnn_type basic_rnn --input_file data/shakespeare.txt --seed "KING LEAR" --sample_len 500
```

LSTM (two recurrent layers, 192 hidden units):

```bash
python text_generation.py --rnn_type lstm_rnn --input_file data/shakespeare.txt --seed "KING LEAR" --sample_len 500
```

To train on the supplied sonnets instead, use `--input_file data/shakespeare_sonnets.txt` and choose a seed that occurs in that corpus, such as `FROM`. **Training is computationally expensive**, particularly with the custom Python time-step loop. The scripts train from scratch when invoked; pretrained checkpoints are not bundled.

The entry point saves a `.pt` state dictionary and a generated `.txt` file with a name derived from `--rnn_type`. Loss and accuracy plots are displayed during training. The historical examples in `examples/` have not been regenerated or modified.

## Quick checks (without retraining)

```bash
python -m unittest discover -s tests -v
```

The tests exercise tensor shapes, gradients, sequence preparation, and prompt-based text generation with randomly initialized toy models. **Passing these tests is not equivalent to reproducing the original full training experiment.**

## Example outputs

See `examples/basic_rnn_output.txt`, `examples/lstm_rnn_output.txt`, and `examples/lstm_additional_sample.txt`. They illustrate the characteristic, sometimes incoherent, outputs of character-level recurrent models. These are not Shakespeare quotations or proof of literary understanding.

## Notes on provenance and code changes

- The project was extracted from the supplied `deep_learning.zip` archive's `Project3/` directory; Projects 1 and 2 and an unrelated tweets dataset were intentionally excluded.
- Source for recurrent cells, model stacking, sequence preparation, and training was preserved from the archive.
- The entry point's **default corpus path** was changed to `data/shakespeare.txt` for the repository layout.
- Sampling was corrected to warm the hidden state with the actual prompt (without appending sampled tokens during prompt processing) and to disable gradient tracking during inference. Original sample outputs were preserved unchanged.
- Incomplete assignment starter scripts and the original disclosure were retained under `archive/` and `docs/` for transparency. Use `text_generation.py`, not the incomplete starter scripts.
- The original file `hw3_llm_and_discussion_disclosure.md` has an unfilled student-name placeholder. Its inclusion documents the source archive; it does not establish authorship. Review provenance and redistribution permissions before claiming this work as your own or publishing it.
- The exact provenance and licensing of the compiled training text are not documented in the archive; verify before redistributing beyond an educational/portfolio context.

## Portfolio description

*Custom character-level RNN and LSTM models in PyTorch for Shakespeare-themed text generation, featuring manual recurrent cells, sequence training, validation, and top-k character sampling.*
