import io
from typing import Any, Dict, List, Optional, Union, Callable
import re
import logging
from huggingface_hub.utils import get_session
from .local_model_interface import LocalModelInterface
from transformers import GenerationConfig
from griptape.utils import PromptStack

logger = logging.getLogger(__name__)

ALL_TASKS = [
    # NLP
    "chat"
]

class LocalOpenOrcaInferenceInvoke(LocalModelInterface):
    """
        Client for invoking a locally installed model for inference.
        The model instance and the name of the generation function are passed as parameters in order to allow some flexibility in expanding beyond the initial target of llama2
    
    Example:

    ```python
    >>> from griptape.prompt import LocalLlamInferenceClient

    >>> # Chat
    >>> inference = LocalLlamaInferenceClient(
                model = LLama
                task='chat',
                gpu=True,
                )

    >>> __call__([[{"role":"user","content":"Who was the first woman to swim the atlantic ocean"}]])
    
    raw_response == True:
        {"data":[{"generation":{"role":"assistant","content":" Oh, that\'s a great question! *adjusts glasses* The first woman ..."}}]}
    else
        " Oh, that\'s a great question! *adjusts glasses* The first woman ..."   
    ```
    """

    def __init__(
        self,
        model: Callable = None,
        model_name: str = '',
        tokenizer: Any = None, # Tokenize instance
        task: str = 'chat',
        gpu: bool = False,  #TODO - remove and assume that the model is already on the right device.
    ):
        """
        Args:
            model(``nn.Module``):
                pytorch model that exposes a chat_completion function
            model_name: (``str``)
                Human readable name of the model.
            task (``str``, defaults ``chat``):
                the name of the task.
            gpu (`bool`, `optional`, defaults `False`):
                Whether to use GPU instead of CPU for inference(requires Startup
                plan at least).
        """

        self.DEFAULT_PARAMS = {
                            'max_new_tokens':512,
                            'max_length':2000,
                            'temperature':0.7
                            }

        self.options = {"wait_for_model": True, "use_gpu": gpu}
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.inference_function_name = 'generate'

        if not hasattr(model, self.inference_function_name):
            raise RuntimeError(f'Model does not have a function called {self.inference_function_name}')
        
    def __repr__(self):
        # Do not add headers to repr to avoid leaking token.
        return f"OpenOrcaInferenceCall(model='{self.model_name()}')"

    def __call__(
        self,
        inputs: Optional[Union[str, Dict, List[str], List[List[str]]]] = None,
        params: Optional[Dict] = None,
        raw_response: bool = False,
    ) -> Any:
        """Generate text.

        Args:
            inputs (`Union[List[str]|List[List[str]]`):
                Inputs for the prediction. Truncated to avoid OOM and converted to dialog format.
            params (`Dict`, *optional*):
                Additional parameters for the models. Will be sent as `parameters` in the
                payload.
            raw_response (`bool`, defaults to `False`):
                If `True`, the raw `Response` object is returned. You can parse its content
                as preferred. By default, the content is parsed into a more practical format
                (json dictionary or PIL Image for example).
        """
        device = "cuda" if self.options['use_gpu'] else "cpu"

        generation_config = GenerationConfig(
                max_length=(params | self.DEFAULT_PARAMS)["max_length"],
                max_new_tokens = (params | self.DEFAULT_PARAMS)["max_new_tokens"],
                temperature=(params | self.DEFAULT_PARAMS)["temperature"],
                top_p=0.95, 
                repetition_penalty=1.0,
                do_sample=True, 
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id, 
                pad_token_id=self.tokenizer.eos_token_id,
                transformers_version="4.34.0.dev0"
            )
        
        if isinstance(inputs, list) and isinstance(inputs[0],list):
            inputs=inputs[0]


        tokenized_inputs = self.tokenizer('\n'.join(inputs), return_tensors='pt', return_attention_mask=True).to(device)
        #inputs = inputs.to(dtype=self.model.dtype).to(device)

        # Call inference on the model.
        response = getattr(self.model, self.inference_function_name)(**tokenized_inputs, generation_config=generation_config)

        # Let the user handle the response
        if raw_response:
            return response
        else:
            # Pull out the returned role and concontent
            # Copy the schema of API call to the same.
            generated_response = self.tokenizer.batch_decode(response)[0]
            pattern = r'<\|im_end\|>'
            generated_response = re.split(pattern, generated_response, re.DOTALL)
            generated_response = generated_response[-2].strip().strip('<|im_start|>').strip().strip(PromptStack.ASSISTANT_ROLE) # the sequence ends with <|im_end|>

            return {'data':[{'generation':{'role':PromptStack.ASSISTANT_ROLE, 'content':generated_response}}]}
