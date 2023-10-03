from os import environ

from griptape.utils import PromptStack

environ["TRANSFORMERS_VERBOSITY"] = "error"

from attr import define, field, Factory

from transformers import AutoTokenizer
from griptape.artifacts import TextArtifact
from griptape.drivers import BasePromptDriver

from griptape.drivers.prompt.local_llama_InferenceAPI import LocalLlamaInferenceApi
from griptape.tokenizers.local_llama_tokenizer import LocalLlamaTokenizer

import asyncio
import aiohttp


@define
class LocalLlamaPromptDriver(BasePromptDriver):
    """
    Attributes:
        inference_endpoint: Base URL of the inference endpoint
        task: e.g. chat
        use_gpu: Use GPU during model run.
        params: Custom model run parameters. 
        client: LocalLlamaInferenceAPI
        tokenizer: Custom `HuggingFaceTokenizer`.
        
    """
    SUPPORTED_TASKS = ["chat"]
    MAX_NEW_TOKENS = 512
    DEFAULT_PARAMS = {
        "return_full_text": False,
        "max_new_tokens": MAX_NEW_TOKENS
    }

    inference_endpoint: str = field(kw_only=True)
    task: str = field(default=True, kw_only=True)
    params: dict = field(factory=dict, kw_only=True)
    use_gpu: bool = field(default=False, kw_only=True)
    async_session: aiohttp.ClientSession = field(kw_only=True)
    tokenizer_path: str = field(default=None, kw_only=True)
    max_tokens: int = field(default =256, kw_only=True)
    model: str = field(default='locallama', kw_only=True)
    
    client: LocalLlamaInferenceApi = field(
        default=Factory(
            lambda self: LocalLlamaInferenceApi(
                inference_endpoint=self.inference_endpoint,
                task='chat',
                gpu=self.use_gpu,
                async_session=self.async_session,
            ),
            takes_self=True
        ),
        kw_only=True
    )

    tokenizer: LocalLlamaTokenizer = field(
        default=Factory(
            lambda self: LocalLlamaTokenizer(
                model_path=self.tokenizer_path
            ), takes_self=True
        ),
        kw_only=True
    )

    def prompt_stack_to_dialog(self, prompt_stack: PromptStack) -> str:
        prompt_lines = []

        for i in prompt_stack.inputs:
            if i.is_assistant():
                prompt_lines.append({"role":"assistant", "content": f"{i.content}"})
            elif i.is_user():
                prompt_lines.append({"role":"user", "content": f"{i.content}"})
            else:
                prompt_lines.append({"role":"system", "content": f"{i.content}"})

        #prompt_lines.append("Assistant:")

        return [prompt_lines]


    def try_run(self, prompt_stack: PromptStack) -> TextArtifact:
        prompt = self.prompt_stack_to_string(prompt_stack)
        dialog = self.prompt_stack_to_dialog(prompt_stack)

        if self.client.task in self.SUPPORTED_TASKS:
            response = self.client(
                inputs=dialog,
                params=self.DEFAULT_PARAMS | self.params
            )

            if len(response) == 1:
                return TextArtifact(
                    #value=response[0]["generated_text"].strip()
                    value = response['data'][0]['generation']['content']
                )
            else:
                raise Exception("Completion with more than one choice is not supported yet.")
        else:
            raise Exception(f"Only models with the following tasks are supported: {self.SUPPORTED_TASKS}")
