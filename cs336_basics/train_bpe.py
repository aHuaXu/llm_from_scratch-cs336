import multiprocessing
import os
import regex as re
from multiprocessing import Pool
from typing import Generator, Tuple, List, Pattern, Union
from tqdm import tqdm
from cs336_basics.pretokenization_example import (
    find_chunk_boundaries,
    get_expected_chunk_num,
)

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    vocab = {i: bytes([i]) for i in range(256)}

    for token in special_tokens:
        bytes_token = token.encode("utf-8")
        vocab[len(vocab)] = bytes_token
    return vocab


def word2bytes(word: str) -> tuple[bytes, ...]:
    a = list(word.encode("utf-8"))
    return tuple(bytes([b]) for b in a)


def chunk_generator(
    input_path: str | os.PathLike,
    chunk_num: int,
    special_tokens: List[str] | None = None,
    split_consider_memory: bool = False,  # if True split file into expected chunk size
) -> Generator[tuple[str, List[str]], None, None]:
    with open(input_path, "rb") as f:
        if split_consider_memory:
            chunk_num = get_expected_chunk_num(f)
        boundaries = find_chunk_boundaries(f, chunk_num, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk = chunk.replace("\r\n", "\n")  # 兼容windows
            yield chunk, special_tokens


def default_chunk_generator(
    input_path: str | os.PathLike,
    chunk_num: int,
    split_consider_memory: bool = False,  # if True split file into expected chunk size
) -> Generator[str, None, None]:
    for chunk, _ in chunk_generator(
        input_path, chunk_num, split_consider_memory=split_consider_memory
    ):
        yield chunk


def token_from_chunk_generator(
    chunk: str,
    special_tokens: List[str],
    drop_special: bool = True,
) -> Generator[str, None, None]:
    # set default separator token
    special_tokens = sorted(special_tokens, key=len, reverse=True)
    separator = "|".join([re.escape(token) for token in special_tokens])
    if not drop_special:
        separator = f"({separator})"
    separator_pat = re.compile(separator)

    sub_chunks = separator_pat.split(chunk)
    for sub_chunk in sub_chunks:
        if not drop_special and sub_chunk in special_tokens:
            yield sub_chunk
        else:
            for match in PAT.finditer(sub_chunk):
                token = match.group()
                yield token


def count_one_chunk(
    chunk: str,
    special_tokens: List[str],
    drop_special: bool = True,
) -> dict[tuple[bytes, ...], int]:
    pre_token_counts: dict[tuple[bytes, ...], int] = {}
    # Run pre-tokenization on your chunk and store the counts for each pre-token
    for token in token_from_chunk_generator(chunk, special_tokens, drop_special):
        token_tuple = word2bytes(token)
        if len(token_tuple) < 2:
            continue
        pre_token_counts[token_tuple] = pre_token_counts.get(token_tuple, 0) + 1
    return pre_token_counts


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 1. Vocab
    # dict[int, bytes]
    vocab: dict[int, bytes] = init_vocab(special_tokens)

    # dict[tuple[bytes], int]
    pre_token_counts: dict[tuple[bytes, ...], int] = {}

    separator = "|".join([re.escape(token) for token in special_tokens])
    separator_pat = re.compile(separator)

    # 2. Pre-tokenization
    num_processes = (
        2  # test_train_bpe_speed will fail in windows if it's set with 4 ...
    )
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(
            count_one_chunk, chunk_generator(input_path, 4, special_tokens)
        )
        for one_chunk_count in results:
            for token, count in one_chunk_count.items():
                pre_token_counts[token] = pre_token_counts.get(token, 0) + count
    # print("pre_token_counts:{}".format(pre_token_counts))

    # 3. count the initial token counts
    token_pairs_counts: dict[tuple[bytes, bytes], int] = {}
    merges: list[tuple[bytes, bytes]] = []
    for pre_token, count in pre_token_counts.items():
        for pair in zip(pre_token[:-1], pre_token[1:]):
            # (bytes, bytes)
            token_pairs_counts[pair] = token_pairs_counts.get(pair, 0) + count

    # 4. Merges
    # Calculate total merges needed
    total_merges = vocab_size - len(vocab)

    # Create progress bar
    with tqdm(total=total_merges, desc="Training BPE", unit="merge") as pbar:
        while len(vocab) < vocab_size:
            # get the most frequent pair, if there are multiple pairs with the same frequency, choose the lexicographically greater pair
            most_frequent_pair, _ = max(
                token_pairs_counts.items(), key=lambda x: (x[1], x[0])
            )
            # print("most_frequent_pair:{}, token_pairs_counts:{}".format(most_frequent_pair, token_pairs_counts))

            merges.append(most_frequent_pair)
            merged_token = most_frequent_pair[0] + most_frequent_pair[1]
            vocab[len(vocab)] = merged_token

            # replace the pre_token with the merged token
            for pre_token, count in list(pre_token_counts.items()):
                n = len(pre_token)
                if most_frequent_pair not in zip(pre_token[:-1], pre_token[1:]):
                    continue

                # generate the new pre_token
                new_pre_token: list[bytes] = []
                i = 0
                while i < n:
                    # match the most frequent pair
                    if (
                        i < n - 1
                        and pre_token[i] == most_frequent_pair[0]
                        and pre_token[i + 1] == most_frequent_pair[1]
                    ):
                        new_pre_token.append(merged_token)
                        i += 2
                    else:
                        new_pre_token.append(pre_token[i])
                        i += 1

                # clear pre_token pair count
                for pair in zip(pre_token[:-1], pre_token[1:]):
                    token_pairs_counts[pair] -= count
                    if token_pairs_counts[pair] == 0:
                        del token_pairs_counts[pair]
                # count the new_pre_token pair count
                for pair in zip(new_pre_token[:-1], new_pre_token[1:]):
                    token_pairs_counts[pair] = token_pairs_counts.get(pair, 0) + count

                # update the pre_token_counts with the new_pre_token
                pre_token_counts[tuple(new_pre_token)] = count
                del pre_token_counts[pre_token]

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(
                {"vocab_size": len(vocab), "remaining_pairs": len(token_pairs_counts)}
            )

    return vocab, merges


# ToDo: assignment1 #L10
