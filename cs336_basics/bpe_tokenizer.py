from multiprocessing import Pool
from typing import Iterable, Iterator, List
from tqdm import tqdm
from cs336_basics.train_bpe import (
    default_chunk_generator,
    token_from_chunk_generator,
    word2bytes,
    load_vocab_and_merges,
)

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

        self.merges_priority_map: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(self.merges)}

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

    def _get_bpe_merges(self, pre_token: str) -> tuple[bytes, ...]:
        """
        对字节片段进行BPE编码，返回字节列表
        """
        parts: tuple[bytes, ...] = word2bytes(pre_token)
        while len(parts) > 1:
            # 查找所有可能的合并对
            pairs = set()
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i + 1])
                if pair in self.merges_priority_map:
                    pairs.add(pair)

            if not pairs:
                break

            # 找到优先级最高的合并对
            best_pair = min(pairs, key=lambda pair: self.merges_priority_map[pair])

            # 执行合并
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i + 1]) == best_pair:
                    new_parts.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = tuple(new_parts)

        return parts

    def _tokens_to_ids(self, tokens: Iterable[bytes]) -> list[int]:
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
                    token_tuple = self._get_bpe_merges(pre_token)
                    res.extend(self._tokens_to_ids(token_tuple))
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
