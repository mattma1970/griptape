import io
from typing import Any, Dict, List, Optional, Union, Callable
import re
import logging
from huggingface_hub.utils import get_session
from .llama_interface import LlamaInterface

logger = logging.getLogger(__name__)

ALL_TASKS = [
    # NLP
    "chat"
]

class LocalLlamaInferenceInvoke(LlamaInterface):
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

        self.options = {"wait_for_model": True, "use_gpu": gpu}
        self.model = model
        self.task = task
        self.inference_function_name = 'chat_completion' if self.task =='chat' else 'text_completion'

        if not hasattr(model, self.inference_function_name):
            raise RuntimeError('Model does not have a function called chat_completion.')
        
    def __repr__(self):
        # Do not add headers to repr to avoid leaking token.
        return f"LocalLlamaInferenceCall(model='{self.model_name()}')"

    def __call__(
        self,
        inputs: Optional[Union[str, Dict, List[str], List[List[str]]]] = None,
        params: Optional[Dict] = None,
        raw_response: bool = False,
    ) -> Any:
        """Make a call to the Inference API.

        Args:
            inputs (`List[Dict]`):
                Inputs for the prediction.
            params (`Dict`, *optional*):
                Additional parameters for the models. Will be sent as `parameters` in the
                payload.
            raw_response (`bool`, defaults to `False`):
                If `True`, the raw `Response` object is returned. You can parse its content
                as preferred. By default, the content is parsed into a more practical format
                (json dictionary or PIL Image for example).
        """
        
        payload = {"dialogs":inputs,  # List[List[dict]]
                    "max_gen_len":params["max_gen_len"] if "max_gen_len" in params else 512,
                    "temperature":params["temperature"] if "temperature" in params else 0.6,
                    "top_p":params["top_p"] if "top_p" in params else 0.9,
                    }

        # Call inference on the model.
        response = getattr(self.model, self.inference_function_name)(**payload) 

        # Let the user handle the response
        if raw_response:
            return response
        else:
            # Pull out the returned role and concontent
            # Copy the schema of API call to the same.
            return {'data':[{'generation':{'role':response[0]['generation']['role'], 'content':response[0]['generation']['content']}}]}
