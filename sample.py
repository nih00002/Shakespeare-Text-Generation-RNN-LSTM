"""Character-level autoregressive generation for trained RNN/LSTM models."""

import torch
import torch.nn.functional as F


def predict(model, x, h, c, char_to_index, index_to_char, top_k=5):
    """Sample one next character and return the updated recurrent states."""
    if x not in char_to_index:
        raise ValueError(f"Seed includes character absent from the corpus: {x!r}")
    vocabulary_size = len(char_to_index)
    device = next(model.parameters()).device
    token = torch.tensor([char_to_index[x]], device=device)
    encoded = F.one_hot(token, num_classes=vocabulary_size).float().unsqueeze(1)
    with torch.no_grad():
        logits, h, c = model(encoded, h, c)
        probs = torch.softmax(logits[-1], dim=-1)
        if top_k is not None:
            if not 1 <= top_k <= vocabulary_size:
                raise ValueError("top_k must be between 1 and the vocabulary size")
            values, indices = torch.topk(probs, top_k)
            local_choice = torch.multinomial(values, num_samples=1)
            next_index = indices[local_choice].item()
        else:
            next_index = torch.multinomial(probs, num_samples=1).item()
    return index_to_char[next_index], h, c


def sample(model, seed, sample_len, char_to_index, index_to_char,
           keep_seed=True, top_k=5):
    """Warm the recurrent state with the *actual* prompt, then generate sample_len chars.

    The seed is returned in the output when keep_seed=True, but it is not
    mistakenly extended with sampled characters during state warmup.
    """
    if not seed:
        raise ValueError("seed must contain at least one character")
    if sample_len < 0:
        raise ValueError("sample_len must not be negative")
    model.eval()
    device = next(model.parameters()).device
    h, c = model.initial_hidden_states(batch_size=1)
    h, c = h.to(device), c.to(device)

    vocabulary_size = len(char_to_index)
    # Prime the model with each seed character in sequence.
    with torch.no_grad():
        logits = None
        for character in seed:
            if character not in char_to_index:
                raise ValueError(f"Seed includes character absent from the corpus: {character!r}")
            token = torch.tensor([char_to_index[character]], device=device)
            encoded = F.one_hot(token, num_classes=vocabulary_size).float().unsqueeze(1)
            logits, h, c = model(encoded, h, c)

        generated = []
        for _ in range(sample_len):
            probs = torch.softmax(logits[-1], dim=-1)
            if top_k is not None:
                if not 1 <= top_k <= vocabulary_size:
                    raise ValueError("top_k must be between 1 and the vocabulary size")
                values, indices = torch.topk(probs, top_k)
                next_index = indices[torch.multinomial(values, num_samples=1)].item()
            else:
                next_index = torch.multinomial(probs, num_samples=1).item()
            character = index_to_char[next_index]
            generated.append(character)
            token = torch.tensor([next_index], device=device)
            encoded = F.one_hot(token, num_classes=vocabulary_size).float().unsqueeze(1)
            logits, h, c = model(encoded, h, c)
    return (seed if keep_seed else "") + "".join(generated)
