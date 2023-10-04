from os import environ

from griptape.utils import PromptStack

environ["TRANSFORMERS_VERBOSITY"] = "error"

from attr import define, field, Factory

from transformers import AutoTokenizer
from griptape.artifacts import TextArtifact
from griptape.drivers import BasePromptDriver

from griptape.drivers.prompt.model_specific import LocalLlamaInferenceApi, LocalLlamaInferenceCall

from griptape.tokenizers.model_specific.local_llama_tokenizer import LocalLlamaTokenizer
from typing import Any

import re


@define
class LocalLlamaPromptDriver(BasePromptDriver):
    """
    Attributes:
        inference_resource: Base URL of the inference endpoint OR the fully file name of the model file.
        task: e.g. chat
        use_gpu: Use GPU during model run.
        client: LocalLlamaInferenceAPI
        tokenizer_path: os.Pathlike: local tokenizer model file path.
    """
    SUPPORTED_TASKS = ["chat"]
    MAX_NEW_TOKENS = 512
    DEFAULT_PARAMS = {
        "return_full_text": False,
        "max_new_tokens": MAX_NEW_TOKENS
    }

    inference_resource: Any = field(kw_only=True) # can be either a URL or a nn.Module model
    task: str = field(default='chat', kw_only=True)
    params: dict = field(factory=dict, kw_only=True)
    use_gpu: bool = field(default=False, kw_only=True)
    tokenizer_path: str = field(default=None, kw_only=True)
    max_tokens: int = field(default =256, kw_only=True)
    model: str = field(default='locallama', kw_only=True)
    

    if isinstance(inference_resource,str):
        # Assume its a URL
        client: LocalLlamaInferenceApi = field(
            default=Factory(
                lambda self: LocalLlamaInferenceApi(
                    inference_endpoint=self.inference_resource,
                    task='chat',
                    gpu=self.use_gpu,
                ),
                takes_self=True
            ),
            kw_only=True
        )
    else:
        # else its a model
        client: LocalLlamaInferenceCall = field(
            default=Factory(
                lambda self: LocalLlamaInferenceCall(
                    model=self.inference_resource,
                    task ='chat',
                    gpu=self.use_gpu,
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
        '''
        Parse the prompt_stack into the List[List[Dict]] expected by Llama2 local model.
        '''
        prompt_lines = []

        for i in prompt_stack.inputs:
            if i.is_assistant():
                prompt_lines.append({"role":"assistant", "content": f"{i.content}"})
            elif i.is_user():
                prompt_lines.append({"role":"user", "content": f"{i.content}"})
            else:
                prompt_lines.append({"role":"system", "content": f"{i.content}"})

        return [prompt_lines]


    def try_run(self, prompt_stack: PromptStack) -> TextArtifact:
        dialog = self.prompt_stack_to_dialog(prompt_stack)

        if self.client.task in self.SUPPORTED_TASKS:
            response = self.client(
                inputs=dialog,
                params=self.DEFAULT_PARAMS | self.params
            )

            if len(response) == 1:
                return TextArtifact(
                    value = response['data'][0]['generation']['content'].strip()
                )
            else:
                raise Exception("Completion with more than one choice is not supported yet.")
        else:
            raise Exception(f"Only models with the following tasks are supported: {self.SUPPORTED_TASKS}")
