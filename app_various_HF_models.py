import os
from griptape.structures import Agent
from griptape.drivers import HuggingFaceHubPromptDriver, HuggingfaceLlamaPromptDriver, OpenAiChatPromptDriver, BasePromptDriver
from griptape.rules import Rule, Ruleset
from griptape.utils import PromptStack
from griptape.utils import Chat
from griptape.tools import WebScraper, WebSearch
from typing import Callable

from dotenv import load_dotenv

    
def llama2_prompt_stack_to_string(prompt_stack: PromptStack) -> str:
    """ [https://huggingface.co/blog/llama2#inference]
        <s>[INST] <<SYS>>
        {{ system_prompt }}
        <</SYS>>
        {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s><s>[INST] {{ user_msg_2 }} [/INST]
    """

    prompt_lines = []

    for i in prompt_stack.inputs:
        if i.is_user():
            prompt_lines.append(f"{i.content} [/INST] ")
        elif i.is_assistant():
            prompt_lines.append(f"{i.content} </s><s>[INST]")
        else:
            '''System prompt will be the first one in the list if it exists at all.'''
            prompt_lines.append(
                                f"""<s>[INST] <<SYS>>
                                {i.content}
                                <</SYS>>
                                
                                """)

    return "".join(prompt_lines)

def mistral_prompt_stack_to_string(prompt_stack: PromptStack):
    prompt_lines = []
    for i in prompt_stack.inputs:
        if i.is_user():
            prompt_lines.append(f"{i.content} [/INST] ")
        elif i.is_assistant():
            prompt_lines.append(f"{i.content} </s><s>[INST]")
        else:
            prompt_lines.append(
                                f"""<s>[INST] 
                                {i.content}
                                """)

    return "".join(prompt_lines)

def openorca_prompt_stack_to_string(prompt_stack: PromptStack):
    prompt_lines = []
    for i in prompt_stack.inputs:
        if i.is_user():
            prompt_lines.append(f"""<|im_start|>user
                                {i.content} <|im_end|>
                                """)
        elif i.is_assistant():
            prompt_lines.append(f"""<|im_start|>
                                {i.content}<|im_end|>
                                """)
        else:
            prompt_lines.append(
                                f"""<|im_start|>system
                                {i.content}
                                <|im_end|>
                                """)

    return "".join(prompt_lines)


params =  {
            "max_new_tokens": 512, #new tokens per generation
            "temperature":0.7,
            "stop_sequences":['[/INST]'], 
            'stream':True,
        }

def huggingface_api(model: str, prompt_to_string: Callable = None):
    agent = Agent(
    prompt_driver=HuggingFaceHubPromptDriver(
        repo_id = model,
        api_token=os.environ['HUGGING_FACE_API_TOKEN'], params = params, prompt_stack_to_string=prompt_to_string if prompt_to_string else BasePromptDriver.default_prompt_stack_to_string_converter),
        tools=[WebSearch(google_api_key=os.environ['google_api_key'], google_api_search_id=os.environ['google_api_search_id'])],
)
    return agent

def huggingface_llama_api(model: str):
    agent = Agent(
    prompt_driver=HuggingfaceLlamaPromptDriver(
        repo_id = model,
        api_token=os.environ['HUGGING_FACE_API_TOKEN'], params = params),
        tools=[WebSearch(google_api_key=os.environ['google_api_key'], google_api_search_id=os.environ['google_api_search_id'])]
)
    return agent


def openai_api(model = 'gpt-3.5-turbo'):
    agent = Agent(
        prompt_driver=OpenAiChatPromptDriver(
            model=model),
            #tools=[WebSearch(google_api_key=os.environ['google_api_key'], google_api_search_id=os.environ['google_api_search_id'])]
    )
    return agent

#Chat(huggingface_api('Open-Orca/Mistral-7B-OpenOrca', openorca_prompt_stack_to_string)).start()
#Chat(huggingface_api('mistralai/Mistral-7B-Instruct-v0.1', mistral_prompt_stack_to_string)).start()
#Chat(huggingface_api('Open-Orca/LlongOrca-13B-16k', openorca_prompt_stack_to_string)).start()
#Chat(huggingface_api('meta-llama/Llama-2-70b-chat-hf',llama2_prompt_stack_to_string)).start()
Chat(openai_api()).start()