import io
from typing import Any, Dict, List, Optional, Union
import re
import logging
from huggingface_hub.utils import get_session
import asyncio

from huggingface_hub.utils import get_session


logger = logging.getLogger(__name__)


ALL_TASKS = [
    # NLP
    "chat"
]

class LocalLlamaInferenceApi:
    """Client to configure requests and make calls to the HuggingFace Inference API.

    Example:

    ```python
    >>> from huggingface_hub.inference_api import InferenceApi

    >>> # Mask-fill example
    >>> inference = InferenceApi("bert-base-uncased")
    >>> inference(inputs="The goal of life is [MASK].")
    [{'sequence': 'the goal of life is life.', 'score': 0.10933292657136917, 'token': 2166, 'token_str': 'life'}]

    >>> # Question Answering example
    >>> inference = InferenceApi("deepset/roberta-base-squad2")
    >>> inputs = {
    ...     "question": "What's my name?",
    ...     "context": "My name is Clara and I live in Berkeley.",
    ... }
    >>> inference(inputs)
    {'score': 0.9326569437980652, 'start': 11, 'end': 16, 'answer': 'Clara'}

    >>> # Zero-shot example
    >>> inference = InferenceApi("typeform/distilbert-base-uncased-mnli")
    >>> inputs = "Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"
    >>> params = {"candidate_labels": ["refund", "legal", "faq"]}
    >>> inference(inputs, params)
    {'sequence': 'Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!', 'labels': ['refund', 'faq', 'legal'], 'scores': [0.9378499388694763, 0.04914155602455139, 0.013008488342165947]}

    ```
    """

    def __init__(
        self,
        inference_endpoint: str = None,
        task: Optional[str] = 'chat',
        gpu: bool = False,
        async_session: Any = None,
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
            async_session: (`aiohttp.ClientSession`)
                The aiohttp.ClientSession to use for making a non-blocking http call using aiohttp
        """

        self.options = {"wait_for_model": True, "use_gpu": gpu, "use_async":(async_session is not None)}
        self.async_session = async_session
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
        return f"InferenceAPI(end_point='{self.api_url}') use_aiohttp: {self.async_session is not None}"

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
        if self.options['use_async']:
            response = self.async_session.post(url=self.api_url, json = inputs )
        else:
            # Still allow blocking calls for backward compatibility
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