"""
Binary bag-of-words ("token indication") features for the BPE-tokenized SNLI dataset.

Every sentence becomes a vector of length |vocabulary| whose entry j is 1 if BPE token j
occurs at least once in the sentence and 0 otherwise (counts are ignored, order is ignored).

The vocabulary is the BPE vocabulary fitted on the training split by snli_bpe.py, so no
information from validation/test leaks into the feature space. Tokens that are not in the
vocabulary are mapped to the model's UNK symbol.

Usage (from the repository root):
    python -m project1.part1.snli_bow --vocab-size 1000 --output-dir project1/results/snli_bow

Or from Python / a notebook:
    from project1.part1 import snli_bow
    model, vocab, features = snli_bow.bow_snli(vocab_size=1000)
    X = features["train"]["premise"]          # scipy.sparse CSR matrix, shape (n_pairs, len(vocab))
    y = features["train"]["label"]            # numpy array, 0=entailment 1=neutral 2=contradiction
    X_pair = snli_bow.pair_features(features["train"])   # [premise | hypothesis], (n_pairs, 2 * len(vocab))
"""
import argparse
import json
import os

import numpy as np
from scipy import sparse

from . import snli_bpe


def build_token_index(model):
    """
    Map every BPE symbol in the model's vocabulary to a column index.
    Symbols are sorted so the mapping is deterministic.
    Returns (token_to_idx, idx_to_token).
    """
    idx_to_token = sorted(model["vocabulary"])
    if model["UNK"] not in model["vocabulary"]:
        idx_to_token.append(model["UNK"])
    token_to_idx = {token: i for i, token in enumerate(idx_to_token)}
    return token_to_idx, idx_to_token


def binary_bow(token_lists, token_to_idx, unk):
    """
    Turn a list of token lists into a binary (0/1) sparse matrix.

    Entry [i, j] is 1 iff token j occurs at least once in token_lists[i].
    Tokens outside the index are mapped to `unk`.
    Returns a scipy.sparse.csr_matrix of dtype uint8, shape (len(token_lists), len(token_to_idx)).
    """
    unk_idx = token_to_idx[unk]
    indptr = [0]
    indices = []
    for tokens in token_lists:
        present = {token_to_idx.get(token, unk_idx) for token in tokens}  # a set: occurrence only
        indices.extend(sorted(present))
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.uint8)
    return sparse.csr_matrix(
        (data, np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int64)),
        shape=(len(token_lists), len(token_to_idx)),
    )


def featurize_split(rows, token_to_idx, unk):
    """Binary BoW matrices for the premises and hypotheses of one tokenized split, plus labels."""
    return {
        "premise": binary_bow([r["premise_tokens"] for r in rows], token_to_idx, unk),
        "hypothesis": binary_bow([r["hypothesis_tokens"] for r in rows], token_to_idx, unk),
        "label": np.array([r["label"] for r in rows], dtype=np.int64),
    }


def pair_features(split_features):
    """Concatenate premise and hypothesis vectors side by side: shape (n_pairs, 2 * vocab_size)."""
    return sparse.hstack(
        [split_features["premise"], split_features["hypothesis"]], format="csr"
    )


def bow_snli(vocab_size=1000, data=None):
    """
    Fit BPE on SNLI train, tokenize every split and build binary bag-of-words features.

    Returns (model, idx_to_token, features) where features maps split name to
    {"premise": csr, "hypothesis": csr, "label": ndarray}.
    """
    model, tokenized = snli_bpe.tokenize_snli(vocab_size, data=data)
    token_to_idx, idx_to_token = build_token_index(model)
    features = {
        split: featurize_split(rows, token_to_idx, model["UNK"])
        for split, rows in tokenized.items()
    }
    return model, idx_to_token, features


def save_features(features, idx_to_token, output_dir):
    """Write vocabulary.json and one <split>_{premise,hypothesis}.npz + <split>_labels.npy per split."""
    with open(os.path.join(output_dir, "vocabulary.json"), "w", encoding="utf-8") as f:
        json.dump(idx_to_token, f, ensure_ascii=False, indent=1)
    for split, feats in features.items():
        sparse.save_npz(os.path.join(output_dir, f"{split}_premise.npz"), feats["premise"])
        sparse.save_npz(os.path.join(output_dir, f"{split}_hypothesis.npz"), feats["hypothesis"])
        np.save(os.path.join(output_dir, f"{split}_labels.npy"), feats["label"])


def main():
    parser = argparse.ArgumentParser(description="Binary bag-of-words features for BPE-tokenized SNLI.")
    parser.add_argument("--vocab-size", type=int, default=1000,
                        help="Maximum BPE vocabulary size, including characters and UNK (default: 1000).")
    parser.add_argument("--output-dir", default=os.path.join("project1", "results", "snli_bow"),
                        help="Where to write vocabulary.json and the feature files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Fitting BPE (vocab size <= {args.vocab_size}) and building binary BoW features ...")
    model, idx_to_token, features = bow_snli(args.vocab_size)
    print(f"  feature dimension per sentence: {len(idx_to_token)}")

    for split, feats in features.items():
        X = feats["premise"]
        density = X.nnz / (X.shape[0] * X.shape[1])
        print(f"  {split}: {X.shape[0]} pairs, premise density {density:.3%}")

    save_features(features, idx_to_token, args.output_dir)

    # Sanity check on the first training example
    first = features["train"]
    active = [idx_to_token[j] for j in first["premise"][0].indices]
    print("\nExample premise active tokens:", active)
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()