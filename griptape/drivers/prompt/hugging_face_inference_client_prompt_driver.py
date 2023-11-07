from typing import Iterator, Optional, Dict, Callable
from os import environ

from griptape.utils import PromptStack

environ["TRANSFORMERS_VERBOSITY"] = "error"

import attr
from attr import define, field, Factory, has
from huggingface_hub import InferenceApi
from huggingface_hub import InferenceClient

from transformers import AutoTokenizer
from griptape.artifacts import TextArtifact
from griptape.drivers import BasePromptDriver
from griptape.tokenizers import HuggingFaceTokenizer

@define(kw_only=True)
class ExtendedInferenceClient(InferenceClient):
    """
    Extends InferenceClient with attributes used with InferenceAPI
    Additional attributes:
        task: The SUPPORTED_TASK
    """
    model: str = field() 
    pretrained_tokenizer: str =field()
    token: str = field(default='NONE',kw_only=True) # Only needed if using Huggingface infrastructure
    timeout: Optional[float] = field(default=None)
    headers: Optional[Dict[str,str]] = field(default={"Content-Type": "application/json"})
    cookies: Optional[Dict[str,str]] = field(default=None)
    task: str = field(default='text_generation')

    def __attr_post_init__(self):
        self.token=self.api_token


@define
class HuggingFaceInferenceClientPromptDriver(BasePromptDriver):
    """
    Attributes:
        model: Hugging Face Hub model name or endpoint URL https://huggingface.co/docs/huggingface_hub/package_reference/inference_client
        pretrained_tokenizer: Specify with Hugging Face Hub model name or the path to the locally saved Autotokenizer.
        token: Hugging Face Hub API token.
        timeout, headers, cookiers: InferenceClient parameters.
        params: InferenceClient task specific parameters e.g. for text_generation see https://huggingface.co/docs/huggingface_hub/v0.18.0.rc0/en/package_reference/inference_client#huggingface_hub.inference._text_generation.TextGenerationParameters
        
        task: The SUPPORTED_TASK that should be invoked on the InferenceClient.
        stream: bool indicating if endpoint should be stream responses back.
        stream_chunk_size: Number of chunks of a streaming response to accumulate before yeilding the results.

        client: InferenceClient - Superceeds InferernceAPI.
    """

    SUPPORTED_TASKS = ['text_generation']
    # Defaults for InferenceClient.params
    DEFAULT_PARAMS = {
        'return_full_text': False,
        'max_new_tokens': 1024,
        'temperature': 0.9,
        'stream': False
    }
     # InferernceClient params
    model: str = field(kw_only=True) 
    pretrained_tokenizer: str =field(kw_only=True)
    token: str = field(kw_only=True)
    timeout: Optional[float] = field(default=None, kw_only=True)
    headers: Optional[Dict[str,str]] = field(default=Factory(dict), kw_only=True)
    cookies: Optional[Dict[str,str]] = field(default=None, kw_only=True)

    params: dict = field(
        default=Factory(dict),
            kw_only=True,
    ) #InferenceClient task specific parameters

    task: Optional[str] = field(default='text_generation',kw_only=True)
    stream: Optional[bool] = field(default=False, kw_only=True)
    stream_chunk_size: Optional[int] = field(default=5, kw_only=True)

    client: ExtendedInferenceClient= field(
        default=Factory(
            lambda self: ExtendedInferenceClient(
                model=self.model, 
                pretrained_tokenizer=self.pretrained_tokenizer,
                token=self.token,
                timeout=self.timeout,
                headers = self.headers,
                cookies = self.cookies, 
                task = self.task,
            ),
            takes_self=True,
        ),
        kw_only=True,
    )
    tokenizer: HuggingFaceTokenizer = field(
        default=Factory(
            lambda self: HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(self.pretrained_tokenizer),
                max_tokens= (self.DEFAULT_PARAMS | self.params)['max_new_tokens'],
            ),
            takes_self=True,
        ),
        kw_only=True,
    )
    prompt_stack_to_string: Callable[[PromptStack], str] = field(
        default = Factory(
            lambda self: self.apply_chat_template(self.tokenizer.tokenizer.apply_chat_template, add_generation_prompt=True, tokenize=False),
            takes_self=True,
        ),
        kw_only=True,
    ) 
    
    def __attr_post_init__(self):
        assert self.stream != self.paras['stream'], 'Stream setting in driver must equal params["stream"]'

    def apply_chat_template(self,template_func: Callable, data_field: str = 'inputs', **kwargs: Dict) -> Callable:
        def inner(prompt_stack: PromptStack)-> str:
            inputs = getattr(prompt_stack,data_field)
            return template_func(inputs, **kwargs)
        return inner

    def try_run(self, prompt_stack: PromptStack) -> TextArtifact:
        prompt = self.prompt_stack_to_string(prompt_stack)

        if self.client.task in self.SUPPORTED_TASKS:
            response = getattr(self.client, self.task)(
                prompt, **(self.params | self.DEFAULT_PARAMS))

            if len(response) == 1:
                value=response[0]["generated_text"].strip()
            elif isinstance(response, str):
                value = response.strip()       
            else:
                raise Exception(
                    "completion with more than one choice is not supported yet"
                )
            return TextArtifact(value)
        else:
            raise Exception(
                f"only models with the following tasks are supported: {self.SUPPORTED_TASKS}"
            )

    def try_stream(self, prompt_stack: PromptStack) -> Iterator[TextArtifact]:
        prompt = self.prompt_stack_to_string(prompt_stack)

        if self.client.task in self.SUPPORTED_TASKS:
            result = getattr(self.client, self.task)(
                prompt, **(self.params | self.DEFAULT_PARAMS)
            )
        chunks_counter = 0
        delta_content = ""
        for chunk in result:
            delta_content += chunk
            chunks_counter+=1
            if chunks_counter>self.stream_chunk_size:
                yield TextArtifact(value=delta_content)
                chunks_counter=0
                delta_content = ""
        if chunks_counter>0:
            yield TextArtifact(value=delta_content)
