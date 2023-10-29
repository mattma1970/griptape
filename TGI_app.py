from vllm import LLM, SamplingParams
from dotenv import load_dotenv
import openai
import pprint
import requests
from griptape.structures import Agent
from griptape.events import CompletionChunkEvent, FinishStructureRunEvent

import logging

from griptape.tokenizers import HuggingFaceTokenizer
from griptape.tools import WebSearch
from griptape.utils import Chat, PromptStack
from transformers import AutoTokenizer
from huggingface_hub import InferenceApi
import os


from griptape.drivers import OpenAiChatPromptDriver, HuggingFaceHubPromptDriver, HuggingFaceInferenceClientPromptDriver


def streaming_print(event: CompletionChunkEvent)->None:
    print(event.token,end='')

callbacks = {
        CompletionChunkEvent:[streaming_print],
        FinishStructureRunEvent: [lambda _: print('\n')]
        }

streaming_response = True

agent = Agent(prompt_driver = HuggingFaceInferenceClientPromptDriver(
                model = 'http://localhost:8080/',
                pretrained_tokenizer = '/home/mtman/Documents/Repos/griptape_experiments/upstream-griptape/data/Mistral-7B-OpenOrca',
                max_tokens=1024,
                temperature=1.1,
                params={"stop_sequences":['</s>','<s>','<|im_end|>','<|im_start|>'],"max_new_tokens":1024}, #}, "stream":True}, #?? Sampling parameters as enumerated in SampleParameters(),
                token='DUMMY',
                stream=streaming_response,
                stream_chunk_size=5,
                ),
                event_listeners=callbacks,
                logger_level=logging.ERROR,
                #tools = [WebSearch(google_api_key=os.environ['google_api_key'], google_api_search_id=os.environ['google_api_search_id'])],
        )

if getattr(agent.prompt_driver,'stream'):
    output_fn = lambda x: x
else:
    output_fn = lambda x: print(x)


Chat(agent, output_fn=output_fn).start()

