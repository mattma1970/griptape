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
    Extends InferenceClient to unify API with other griptape clients
    Attributes:
        task: key of the SUPPORTED_TASK dictionary
    """
    token: str = field(default='DUMMY',kw_only=True)
    model: str = field(kw_only=True) # HF namespace/repo_id or URL of endpoint https://huggingface.co/docs/huggingface_hub/package_reference/inference_client
    pretrained_tokenizer: str =field(kw_only=True) # Either HF namespace/repo or local folder where tokenizer is stored.
    timeout: Optional[float] = field(default=None)
    headers: Optional[Dict[str,str]] = field(default=None)
    cookies: Optional[Dict[str,str]] = field(default=None)
    task: Optional[str] = field(default='text-generation')
    use_gpu: Optional[bool] = field(default=False)

@define
class HuggingFaceInferenceClientPromptDriver(BasePromptDriver):
    """
    Attributes:
        token: Hugging Face Hub API token.
        params: Custom model run parameters.
        model: Hugging Face Hub model name or endpoint URL https://huggingface.co/docs/huggingface_hub/package_reference/inference_client
        client: InferenceClient - Superceeds InferernceAPI.
        pretrained_tokenizer: Custom `HuggingFaceTokenizer`. Specify with Hugging Face Hub model name or the path to the locally saved Autotokenizer.
        task: Key of the SUPPORTED_TASK dictionary that should be invoked on the InferenceClient. InferenceClient exposes many functions for specialised tasks such as automatic_speech_recognition, audio_classification.
        use_gpu: For backward compatibility with InferenceAPI
        stream_chunk_size: Number of chunks of a streaming response to accumulate before yeilding the results.

    """

    SUPPORTED_TASKS = {'text-generation':"text_generation"}
    MAX_NEW_TOKENS = 250
    DEFAULT_PARAMS = {
        "return_full_text": True,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    token: str = field(default='NONE',kw_only=True)
    params: dict = field(factory=dict, kw_only=True)
    model: str = field(kw_only=True) 
    pretrained_tokenizer: str =field(kw_only=True) 
    timeout: Optional[float] = field(default=None,kw_only=True)
    headers: Optional[Dict[str,str]] = field(default={"Content-Type": "application/json"}, kw_only=True)
    cookies: Optional[Dict[str,str]] = field(default=None,kw_only=True)
    task: Optional[str] = field(default='text-generation', kw_only=True)
    use_gpu: Optional[bool] = field(default=False,kw_only=True)

    client: ExtendedInferenceClient= field(
        default=Factory(
            lambda self: ExtendedInferenceClient(
                model=self.model, 
                pretrained_tokenizer=self.pretrained_tokenizer,
                token=self.token,
                timeout=self.timeout,
                headers = self.headers,
                cookies=self.cookies, 
                task=self.task,
                use_gpu=self.use_gpu,
            ),
            takes_self=True,
        ),
        kw_only=True,
    )
    tokenizer: HuggingFaceTokenizer = field(
        default=Factory(
            lambda self: HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(self.pretrained_tokenizer),
                max_tokens=self.MAX_NEW_TOKENS,
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
    stream: bool = field(default=False, kw_only=True)
    stream_chunk_size: int = field(default=5, kw_only=True)
    
    def apply_chat_template(self,template_func: Callable, data_field: str = 'inputs', **kwargs: Dict) -> Callable:
        def inner(prompt_stack: PromptStack)-> str:
            inputs = getattr(prompt_stack,data_field)
            return template_func(inputs, **kwargs)
        return inner

    def try_run(self, prompt_stack: PromptStack) -> TextArtifact:
        prompt = self.prompt_stack_to_string(prompt_stack)

        if self.client.task in self.SUPPORTED_TASKS.keys():
            response = getattr(self.client, self.SUPPORTED_TASKS[self.task])(
                prompt, **(self.DEFAULT_PARAMS | self.params)
            )

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

        if self.client.task in self.SUPPORTED_TASKS.keys():
            result = getattr(self.client, self.SUPPORTED_TASKS[self.task])(
                prompt, **(self.DEFAULT_PARAMS | self.params), stream=True
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
