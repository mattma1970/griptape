from os import environ
from logging import getLogger

from attr import define, field, Factory
#from griptape.tokenizers import BaseTokenizer
from griptape.tokenizers.base_tokenizer import BaseTokenizer
from sentencepiece import SentencePieceProcessor

import os
from typing import List

import inspect, types

logger=getLogger()

@define
class OpenOrcaTokenizer(BaseTokenizer):
    
    tokenizer = field(kw_only=False)
    model_max_length = field(default=2000, kw_only=True)

    def Factory(self):
        """
        Add all the methods and public attributes from OpenOrcaTokenizer to the passed in tokenizer which will be an object created using transformer.Autotokenizer.from_pretrained
        """
        _skip_=['tokenizer','Factory','encode','decode']


        for cls in inspect.getmro(OpenOrcaTokenizer):
            for name, value in inspect.getmembers(cls):
                if (inspect.isfunction(value) and not name.startswith("__")) and not name in _skip_:
                    setattr(self.tokenizer,name, types.MethodType(value, self.tokenizer))
        self.tokenizer.cls_token=' '
        self.tokenizer.pad_token=' '
        self.tokenizer.sep_token=' '
        self.tokenizer.mask_token=' '

        return self.tokenizer

    def max_tokens(self):
        return self.model_max_length

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)
    
    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)
    
    