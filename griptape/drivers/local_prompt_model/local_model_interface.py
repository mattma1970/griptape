from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, Any, List

class LocalModelInterface(ABC):
    @abstractmethod
    def __init__(self,inference_endpoint: str = None, task: Optional[str] = 'chat', gpu: bool = False, ):
        ...

    def __call__(self, inputs: Optional[Union[str, Dict, List[str], List[List[str]]]] = None, params: Optional[Dict] = None, data: Optional[bytes] = None,raw_response: bool = False,) -> Any:
        ...
