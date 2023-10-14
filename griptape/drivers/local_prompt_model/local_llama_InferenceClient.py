import io
from typing import Any, Dict, List, Optional, Union
import re
import logging
from huggingface_hub.utils import get_session
from .local_model_interface import LocalModelInterface

logger = logging.getLogger(__name__)

ALL_TASKS = [
    # NLP
    "chat"
]


class LocalLlamaInferenceClient(LocalModelInterface):
    """An inference API client for a llama2 model served locally e.g. via FastAPI
        It allows the base URL of the endpoint to passed in a parameter

    Example:

    ```python
    >>> from griptape.prompt import LocalLlamaInferenceApi

    >>> # Chat
    >>> inference = LocalLlamaInferenceApi(
                inference_endpoint='http://localhost:8080',
                task='chat',
                gpu=True,
                )

    >>> inference([[{"role":"user","content":"Who was the first woman to swim the atlantic ocean"}]])
    
    raw_response == True:
        {"data":[{"generation":{"role":"assistant","content":" Oh, that\'s a great question! *adjusts glasses* The first woman ..."}}]}
    else
        " Oh, that\'s a great question! *adjusts glasses* The first woman ..."   
    ```
    """

    def __init__(
        self,
        inference_endpoint: str = None,
        task: Optional[str] = 'chat',
        gpu: bool = False,
    ):
        """Inits headers and API call information.

        Args:
            end_point_url(``str``):
                rest endpoint: str: e.g. https://localhost:8501/chat
            task (``str``, `optional`, defaults ``chat``):
                LLM task. 
            gpu (`bool`, `optional`, defaults `False`):
                Whether to use GPU instead of CPU for inference(requires Startup
                plan at least).
        """

        self.options = {"use_gpu": gpu}
        self.task = task

        if self.task not in ALL_TASKS:
            raise RuntimeError(f'task "{task} not recognized')

        # check url propertly formed.
        pattern = r"^(http|https):\/\/[^\s/$.?#].[^\s]*$"
        if not re.match(pattern, inference_endpoint):
            raise RuntimeError('Inference URL isn''t validly formed.')
        else:          
            self.api_url = f"{inference_endpoint}/{self.task}"

    def __repr__(self):
        # Do not add headers to repr to avoid leaking token.
        return f"LocalLlamaInferenceAPI(end_point='{self.api_url}')"

    def __call__(
        self,
        inputs: Optional[Union[str, Dict, List[str], List[List[str]]]] = None,
        params: Optional[Dict] = None,
        data: Optional[bytes] = None,
        raw_response: bool = False,
    ) -> Any:
        """Make a call to the Inference API.

        Args:
            inputs (`List[Dict]`):
                Inputs for the prediction.
            params (`Dict`, *optional*):
                Additional parameters for the models. Will be sent as `parameters` in the
                payload.
            data (`bytes`, *optional*):
                Bytes content of the request. In this case, leave `inputs` and `params` empty.
            raw_response (`bool`, defaults to `False`):
                If `True`, the raw `Response` object is returned. You can parse its content
                as preferred. By default, the content is parsed into a more practical format
                (json dictionary or PIL Image for example).
        """
        # Build payload
        payload: Dict[str, Any] = {
            "options": self.options,
        }
        if inputs:
            payload["dialogs"] = inputs #List[List[Dict]]
        if params:
            payload["parameters"] = params

        # Make API call
        response = get_session().post(self.api_url, json=payload)

        # Let the user handle the response
        if raw_response:
            return response

        # By default, parse the response for the user.
        content_type = response.headers.get("Content-Type") or ""
        if content_type == "application/json":
            return response.json()
        else:
            raise NotImplementedError(
                f"{content_type} output type is not implemented yet. You can pass"
                " `raw_response=True` to get the raw `Response` object and parse the"
                " output by yourself."
            )