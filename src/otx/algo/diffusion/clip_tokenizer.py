"""This module provides the ClipTokenizer class for tokenizing text using the ClipTokenizer algorithm."""

from __future__ import annotations

import gzip
import re
from functools import lru_cache

from .utils.download import download


@lru_cache
def _default_bpe() -> str:
    # Clip tokenizer, taken from https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py (MIT license)
    return download(
        "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz",
        ".",
    )


def _get_pairs(word: tuple[str, ...]) -> set:
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    return set(zip(word, word[1:]))


def _whitespace_clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _bytes_to_unicode() -> dict:
    """Returns list of utf-8 byte and a corresponding list of unicode strings.

    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    chars = [chr(n) for n in cs]
    return dict(zip(bs, chars))


class CLIPTokenizer:
    """Tokenizes text using the ClipTokenizer algorithm."""

    def __init__(self) -> None:
        self.byte_encoder = _bytes_to_unicode()
        merges = gzip.open(_default_bpe()).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        bpes = [tuple(merge.split()) for merge in merges]
        vocab = list(_bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in bpes:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(bpes, range(len(bpes))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[^\s]+""", re.IGNORECASE)

    def bpe(self, token: str) -> str:
        """Apply Byte Pair Encoding (BPE) to the given token.

        Args:
            token (str): The token to be encoded.

        Returns:
            str: The encoded token.
        """
        if token in self.cache:
            return self.cache[token]
        word = (*tuple(token[:-1]), token[-1] + "</w>")
        pairs = _get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except Exception:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tuple = tuple(new_word)
            word = new_word_tuple
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        word_str = " ".join(word)
        self.cache[token] = word_str
        return word_str

    def encode(self, text: str, pad_with_zeros: bool = False) -> list[int]:
        """Encode the given text using the ClipTokenizer.

        Args:
            text (str): The text to be encoded.
            pad_with_zeros (bool, optional): Whether to pad the encoded tokens with zeros. Defaults to False.

        Returns:
            List[int]: The encoded tokens.
        """
        bpe_tokens: list[int] = []
        text = _whitespace_clean(text.strip()).lower()
        for token in re.findall(self.pat, text):
            encoded_token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(encoded_token).split(" "))
        # Truncation, keeping two slots for start and end tokens.
        if len(bpe_tokens) > 75:
            bpe_tokens = bpe_tokens[:75]
        return [49406] + bpe_tokens + [49407] + ([0] if pad_with_zeros else [49407]) * (77 - len(bpe_tokens) - 2)
