from os import environ

from griptape.utils import PromptStack

environ["TRANSFORMERS_VERBOSITY"] = "error"

from attr import define, field, Factory

from transformers import AutoTokenizer
from griptape.artifacts import TextArtifact
from griptape.drivers import BasePromptDriver

from griptape.drivers import LocalLlamaInferenceClient, LocalLlamaInferenceInvoke

from griptape.tokenizers.model_specific.local_llama_tokenizer import LocalLlamaTokenizer
from typing import Any, List, Union

import re
import json


@define
class LocalLlamaPromptDriver(BasePromptDriver):
    """
    A driver for locally installed or served Llama2 instance. 
    The appropriate client for either making an API call or a direct invocation of the model for the generation task will be determined based on the inference_resource that is passed.
    Prompt stack slicing is performed to ensure that prompts with fewer than the max_seq_length specified accepted by the model are submitted.

    Attributes:
        inference_resource: Base URL of the inference endpoint OR the fully file name of the model file.
        task: e.g. chat
        use_gpu: Use GPU during model run.
        client: LocalLlamaInferenceAPI
        tokenizer_path: os.Pathlike: local tokenizer model file path.
    """

    SUPPORTED_TASKS = ["chat"]
    MAX_NEW_TOKENS = 128 # the maximum number of token to generate each time. 
    MAX_TOKENS = 1000  # This must be <= max token length specified when building the model ( this is the context window) 
    DEFAULT_PARAMS = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_tokens": MAX_TOKENS,
    }

    inference_resource: Any = field(kw_only=True) # can be either a URL or a nn.Module model
    task: str = field(default='chat', kw_only=True)
    params: dict = field(factory=dict, kw_only=True)
    use_gpu: bool = field(default=False, kw_only=True)   # TODO remove. Llama2 by default is set to use GPUs.
    tokenizer_path: str = field(default=None, kw_only=True)
    model: str = field(default='locallama', kw_only=True)
    
    dialog_tail_length_guess: int = field(init=False, default=1000, kw_only=True,) # An estimate of the length of the tail of the stringified prompt_stack that has less than the max_tokens accepted by the model.

    if isinstance(inference_resource,str):
        # Assume its a URL
        client: LocalLlamaInferenceClient = field(
            default=Factory(
                lambda self: LocalLlamaInferenceClient(
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
        client: LocalLlamaInferenceInvoke = field(
            default=Factory(
                lambda self: LocalLlamaInferenceInvoke(
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

    def prompt_stack_to_dialogs(self, prompt_stack: PromptStack) -> List[List[dict]]:
        '''
        Parse the prompt_stack into the List[List[Dict]] expected by Llama2 local model.
        Instead of converting the entire stake to a string, Dialogs preserved the list structure which is more convenient for manipulation elsewehere.
        @returns: List[List[dict]]: A list of dialogs. 
        '''
        prompt_lines = []

        for i in prompt_stack.inputs:
            if i.is_assistant():
                prompt_lines.append({"role":PromptStack.ASSISTANT_ROLE, "content": f"{i.content}"})
            elif i.is_user():
                prompt_lines.append({"role":PromptStack.USER_ROLE, "content": f"{i.content}"})
            else:
                prompt_lines.append({"role":PromptStack.SYSTEM_ROLE, "content": f"{i.content}"})

        return [prompt_lines]
    
    def dialog_stack_tail(self, prompt_stack: PromptStack, preserve_system_dialog:bool = True) -> List[dict]:
        """
        Get the tail of the prompt_stack that contains fewer the maximum number of tokens the model can accept. This takes into account the maximimum number of 
        tokens that may come from the model response and limits the tail accordingly. i.e max_tokens - max_generated_tokens.

        Args:
            preserve_system_dialog: bool: If Truen, then a system role at the front of the dialog stack should never be removed.
        Returns:
            List[dict]: the tail of the prompt_stack converted into the required dialog format.
        """
        _params: dict =  self.DEFAULT_PARAMS | self.params # use defaults first and overwrite with self.params if provided
        if not 'max_new_tokens' in _params or not 'max_tokens' in _params:
            raise RuntimeError('max_new_tokens or max_tokens param not passed in to local_llama_prompt_driver. These are required and must match those used when instantiating the model.')
        
        dialog_string_stack = [json.dumps(dialog) for dialog in self.prompt_stack_to_dialogs(prompt_stack=prompt_stack)[0]]
        dialog_stack_length = len(dialog_string_stack)
        tail_length = min(dialog_stack_length,self.dialog_tail_length_guess)

        if dialog_stack_length==0:            
            return [] # Empty prompt stack
        
        '''The system dialog is assumed to always be the first dialog in the prompt stack.'''
        system_dialog = ''
        system_dialog_length = 0      

        if preserve_system_dialog and prompt_stack.inputs[0].role == PromptStack.SYSTEM_ROLE:
            system_dialog, dialog_string_stack = dialog_string_stack[0], dialog_string_stack[1:]
            dialog_stack_length-=1
            system_dialog_length = self.dialog_token_count(system_dialog)
        
        cumm_tokens=self.dialog_token_count(dialog_string_stack[-tail_length:])
        max_history_tokens = _params['max_tokens']-_params['max_new_tokens'] - system_dialog_length # Leave room for up to max_new_tokens

        ''' Add or remove dialogs from the tail to keep within the allowed number of historical tokens.'''
        if cumm_tokens<max_history_tokens:
            while cumm_tokens<=max_history_tokens and tail_length<len(dialog_string_stack):
                tail_length=tail_length+1
                cumm_tokens += self.dialog_token_count(dialog_string_stack[-tail_length])
        else:
            while cumm_tokens>max_history_tokens and tail_length>0:
                cumm_tokens -= self.dialog_token_count(dialog_string_stack[-tail_length])
                tail_length -=1

        if tail_length < dialog_stack_length and prompt_stack.inputs[-tail_length].role == PromptStack.ASSISTANT_ROLE:
            tail_length-=1 #llama2 only supports dialogs that for a sequence of dialogs with roles system, user, assistant, user, assistant.
        self.dialog_tail_length_guess = tail_length # save the current length and use as a starting point for next iteration.

        ret = [json.loads(dialog) for dialog in dialog_string_stack[-tail_length:]]
        if preserve_system_dialog and tail_length<dialog_stack_length:  #Only add it back if it was dropped off.
            ret = [json.loads(system_dialog)]+ret
        return ret

    def dialog_token_count(self, dialog_stack: Union[str,List[str]]) -> int:
        '''
        Args:
            dialog_stack: List[str]: list of the stringified dialogs.
        '''
        if isinstance(dialog_stack, str):
            dialog_stack=[dialog_stack]
        return sum([len(self.tokenizer.encode(t)) for t in dialog_stack])
    
    def try_run(self, prompt_stack: PromptStack) -> TextArtifact:

        if self.client.task in self.SUPPORTED_TASKS:
            response = self.client(
                inputs=[self.dialog_stack_tail(prompt_stack)],
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
