from os import environ
from logging import getLogger

from attr import define, field, Factory
#from griptape.tokenizers import BaseTokenizer
from griptape.tokenizers.base_tokenizer import BaseTokenizer
from sentencepiece import SentencePieceProcessor

import os
from typing import List

from inspect import getsource

logger=getLogger()

@define(frozen=True)
class LocalLlamaTokenizer(BaseTokenizer):

    # model path is the only attribute to be initialized.
    model_path: os.PathLike = field(kw_only=True)

    tokenizer = field(init=False, default=Factory(lambda self: SentencePieceProcessor(model_file=self.model_path), takes_self=True),kw_only=False)
    n_words = field(init=False, default = Factory(lambda self: self.tokenizer.vocab_size(), takes_self=True))
    bos_id = field(init=False, default = Factory(lambda self:self.tokenizer.bos_id(), takes_self=True))
    eos_id = field(init=False, default = Factory(lambda self:self.tokenizer.eos_id(), takes_self=True))
    pad_id = field(init=False, default = Factory(lambda self:self.tokenizer.pad_id(), takes_self=True))


    def max_tokens(self):
        return self.tokenizer.model_max_length
    
    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        t = self.tokenizer.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        return self.tokenizer.decode(t)
