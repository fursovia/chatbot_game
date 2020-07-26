from youtokentome import BPE
from transformers import PreTrainedTokenizer
import torch

import typing as tp
from pathlib import Path

PRETRAINED_TOKENIZERS_MAP = {
    "ru-gpt2": Path(__file__).parent / "models" / "ru-gpt2" / "vocab_50000.bpe"
}


class RuGPT2Tokenizer(PreTrainedTokenizer):
    def __init__(self, name_or_vocab_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if name_or_vocab_path in PRETRAINED_TOKENIZERS_MAP:
            name_or_vocab_path = str(PRETRAINED_TOKENIZERS_MAP[name_or_vocab_path])
        self.bpe = BPE(model=name_or_vocab_path)
        self.__vocab = self.get_vocab()

    @classmethod
    def from_pretrained(cls, name_or_vocab_path: str, *inputs, **kwargs):
        return cls(name_or_vocab_path, *inputs, **kwargs)

    def get_vocab(self):
        return self.bpe.vocab()

    def save_vocabulary(self, save_directory):
        pass

    @property
    def bos_token(self):
        return "<|endoftext|>"

    @property
    def bos_token_id(self):
        return 50047

    @property
    def eos_token(self):
        return "<|endoftext|>"

    @property
    def eos_token_id(self):
        return 50047

    @property
    def vocab_size(self):
        return 50048

    def _tokenize(self, text, **kwargs):
        token_ids = self.bpe.encode([text])[0]
        return [self._convert_id_to_token(idx) for idx in token_ids]

    def _convert_token_to_id(self, token: str):
        return self.bpe.encode([token])[0][0]

    def _convert_id_to_token(self, index: int) -> str:
        return self.__vocab[index]

    def decode(self, token_ids, **kwargs) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.bpe.decode(token_ids)

    def encode(self, text: str,
               add_prefix_space: bool = False,
               add_special_tokens: bool = False,
               **kwargs) -> tp.List:
        if add_prefix_space:
            text = " " + text
        add_bos = self.bos_token in text or add_special_tokens
        text = text.replace(self.bos_token, "")
        ids = self.bpe.encode([text], bos=add_bos)[0]
        ids = [idx if idx else self.bos_token_id for idx in ids]
        return ids
