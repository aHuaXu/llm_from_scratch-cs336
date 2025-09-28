from multiprocessing import Pool
from typing import Iterable, Iterator
from tqdm import tqdm
from cs336_basics.train_bpe import (
    default_chunk_generator,
    token_from_chunk_generator,
    word2bytes,
    load_vocab_and_merges,
)


def merge_tuple_pair(
    original: tuple[bytes, ...], target: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    """
    Replace all occurrences of the target (bytes, bytes) subtuple in the original tuple
    with the concatenated result of the two bytes elements (bytes1 + bytes2).

    Args:
        original: The original tuple containing bytes elements (tuple[bytes, ...])
        target: The specific (bytes, bytes) subtuple to be merged

    Returns:
        A new tuple with all target subtuples replaced by their concatenated result
    """
    original_list = list(original)
    result = []
    merged = target[0] + target[1]
    i = 0

    while i < len(original_list):
        if (
            i + 1 < len(original_list)
            and (original_list[i], original_list[i + 1]) == target
        ):
            result.append(merged)
            i += 2
        else:
            result.append(original_list[i])
            i += 1

    return tuple(result)


class BpeTokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Construct a tokenizer from vocabulary and merges."""
        self.vocab = vocab
        self.merges = merges
        if not special_tokens:
            special_tokens = ["<|endoftext|>"]
        self.special_tokens = special_tokens
        self.byte_to_id: dict[bytes, int] = {
            v: k for k, v in self.vocab.items()
        }  # reserve vocab map

        self.supplement_special_tokens(special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """Create a BPE tokenizer from vocabulary and merges files."""
        vocab, merges = load_vocab_and_merges(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def supplement_special_tokens(
        self,
        special_tokens: list[str] | None = None,
    ):
        """Append special_tokens to the vocabulary if they aren’t already there"""
        if special_tokens is None:
            return
        for stoken in special_tokens:
            token = stoken.encode("utf-8")
            if token not in self.byte_to_id:  # 如果不在vocab中，添加它
                num = len(self.vocab)
                self.vocab[num] = token
                self.byte_to_id[token] = num

    def tokens_to_ids(self, tokens: Iterable[bytes]) -> list[int]:
        return [self.byte_to_id[token] for token in tokens]

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        res: list[int] = []
        pre_tokens = list(token_from_chunk_generator(text, self.special_tokens, False))

        with tqdm(
            total=len(pre_tokens), desc="Encoding", unit="token", leave=False
        ) as pbar:
            for pre_token in pre_tokens:
                if pre_token in self.special_tokens:
                    # 特殊token需要转换为bytes
                    special_token_bytes = pre_token.encode("utf-8")
                    res.append(self.byte_to_id[special_token_bytes])
                else:
                    token_tuple = word2bytes(pre_token)
                    for merge_tuple in self.merges:
                        token_tuple = merge_tuple_pair(token_tuple, merge_tuple)
                    res.extend(self.tokens_to_ids(token_tuple))
                pbar.update(1)
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        encoded = b"".join(self.vocab[token_id] for token_id in ids)
        return encoded.decode("utf-8", errors="replace")
