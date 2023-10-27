from typing import Iterator
from os import environ
import requests
import importlib.util
import io

from griptape.utils import PromptStack

environ["TRANSFORMERS_VERBOSITY"] = "error"

from attr import define, field, Factory
from transformers import AutoTokenizer
from griptape.artifacts import TextArtifact
from griptape.drivers import BasePromptDriver
from griptape.tokenizers import HuggingFaceTokenizer
from typing import Optional, Dict, Any, List, Union, Callable

@define 
class AltInferenceApi:
    """Make a call to an inference API where a single HF model is being served. This class is a modification
       to InferenceApi that is used to call the Huggingface_hub Api. This class allows HF models 
       to be served by, for example, vLLM.

    Args:
        api_base (str, defaults to vLLM default http://localhost:8000): 
            URL of the endpoint
        task (str, *optional*):
            chat or completion
        raw_response (`bool`, defaults to `False`):
            If `True`, the raw `Response` object is returned. You can parse its content
            as preferred. By default, the content is parsed into a more practical format
            (json dictionary or PIL Image for example).
    """
    #model: str = field(kw_only=True)
    api_base: str = field(default="http://localhost:8000/", kw_only=True)
    task: Optional[str] = field(default='generate', kw_only=True)
    headers: Optional[Dict] = field(default = None, kw_only=True) # A dictionary of headers to send. Not utilized. Reserved for future authentication information.
    raw_response: bool = field(default=False, kw_only=True)
    api_url = field(init=False)

    def __attrs_post_init__(self):
        self.api_url = f'{self.api_base.rstrip("/")}/{self.task}'

    def __call__(
            self, 
            inputs: Union[str,Dict,List[str],List[List[str]]]=None,
            params: Optional[Dict] = None,
            data: Optional[bytes] = None,
            ):
        """ Args:
            inputs (`str` or `Dict` or `List[str]` or `List[List[str]]`, *optional*):
                Inputs for the prediction.
            params (`Dict`, *optional*):
                Additional parameters for the models. Will be sent as `parameters` in the
                payload.
        """

        # Make API call to an arbitrary URL
        payload = {}

        if inputs:
            payload["prompt"] = inputs
        if params:
            payload.update(**params)

        response = requests.post(self.api_url, headers=self.headers, json=payload, data=data)

        # Let the user handle the response
        if self.raw_response:
            return response

        # By default, parse the response for the user.
        content_type = response.headers.get("Content-Type") or ""
        if content_type.startswith("image"):
            if not importlib.util.find_spec('pillow'):
                raise ImportError(
                    f"Task '{self.task}' returned as image but Pillow is not installed."
                    " Please install it (`pip install Pillow`) or pass"
                    " `raw_response=True` to get the raw `Response` object and parse"
                    " the image by yourself."
                )
            from PIL import Image

            return Image.open(io.BytesIO(response.content))
        elif content_type == "application/json":
            return response.json()
        else:
            raise NotImplementedError(
                f"{content_type} output type is not implemented yet. You can pass"
                " `raw_response=True` to get the raw `Response` object and parse the"
                " output by yourself."
        )


@define
class HuggingFacevLLMPromptDriver(BasePromptDriver):
    """
    Attributes:
        params: Custom model run parameters.
        model: Hugging Face Hub model name. namespace/repo_id
        client: Custom `AltInferenceApi`. A reduced version of HF InferenceAPI designed for calling alternative URLs that serve HF endpoints.
        tokenizer: Custom `HuggingFaceTokenizer`.

    """

    SUPPORTED_TASKS = ["generate"]
    MAX_NEW_TOKENS = 512
    DEFAULT_PARAMS = {
        #"return_full_text": False,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": 0.8,

    }

    #api_token: str = field(kw_only=True)
    api_base: str = field(kw_only=True)
    #use_gpu: bool = field(default=False, kw_only=True)
    params: dict = field(factory=dict, kw_only=True)
    model: str = field(default=None, kw_only=True) # The namespace/repo_id of HF model or the path to the folder where the tokenizer is saved locally.
    client: AltInferenceApi = field(
        default=Factory(
            lambda self: AltInferenceApi(
                api_base='http://localhost:8000/', task='generate', raw_response=False,
            ),
            takes_self=True,
        ),
        kw_only=True,
    )
    tokenizer: HuggingFaceTokenizer = field(
        default=Factory(
            lambda self: HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(self.model),
                max_tokens=self.MAX_NEW_TOKENS,
            ),
            takes_self=True,
        ),
        kw_only=True,
    )

    stream: bool = field(default=False, kw_only=True)

    def __attrs_post_init__(self):
        self.prompt_stack_to_string = self.prompt_stack_to_string_template


    @stream.validator
    def validate_stream(self, _, stream):
        if stream:
            raise ValueError("streaming is not supported")

    def prompt_stack_to_string_template(self, prompt_stack: PromptStack) -> str:
        """
            Use the tokenizer.apply_chat_template to properly format the messages

            prompt_stack (PromptStack)
                messages to be converts to string for prompting.
        """
        ret = None
        try:
            ret = self.tokenizer.tokenizer.apply_chat_template(prompt_stack.inputs, tokenize=False, add_generation_prompt=True)
        except:
            raise NotImplementedError(f'Tokenizer {self.tokenizer.__name__} does not support the apply_chat_template function')
        return ret

    def try_run(self, prompt_stack: PromptStack) -> TextArtifact:
        prompt = self.prompt_stack_to_string(prompt_stack)

        if self.client.task in self.SUPPORTED_TASKS:
            response = self.client(
                inputs=prompt, params=self.DEFAULT_PARAMS | self.params
            )

            if len(response) == 1:
                return TextArtifact(value=response["text"][0].strip())
            else:
                raise Exception(
                    "completion with more than one choice is not supported yet"
                )
        else:
            raise Exception(
                f"only models with the following tasks are supported: {self.SUPPORTED_TASKS}"
            )

    def try_stream(self, _: PromptStack) -> Iterator[TextArtifact]:
        raise NotImplementedError("streaming is not supported")
