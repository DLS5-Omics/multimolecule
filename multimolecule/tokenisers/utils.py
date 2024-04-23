from itertools import product
from typing import List


def generate_kmer_vocabulary(vocabulary: List[str], nmers: int = 1) -> List[str]:
    """
    Generates a kmer vocabulary given an original vocabulary and the size of kmers.

    Args:
        vocabulary (List[str]): The original vocabulary.
        nmers (int, defaults to 1): The size of the kmers to generate.

    Returns:
        vocabulary (List[str]): The kmer vocabulary.
    """

    if nmers <= 1:
        return vocabulary

    special_tokens, tokens = [], []
    for token in vocabulary:
        if token.startswith("<") or token.startswith("["):
            special_tokens.append(token)
        else:
            tokens.append(token)

    tokens = ["".join(kmer) for kmer in product(tokens, repeat=nmers)]

    return special_tokens + tokens
