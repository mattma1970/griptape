import griptape
from griptape.structures import Agent
from griptape.tools import WebSearch
from griptape.drivers import OpenAiChatPromptDriver, HuggingFaceHubPromptDriver
from griptape.utils import Chat

import os
from dotenv import load_dotenv

load_dotenv()

#agent = Agent(prompt_driver=OpenAiChatPromptDriver(model='gpt-4'),
#              tools=[WebSearch(google_api_key=os.environ['google_api_key'], google_api_search_id=os.environ['google_api_search_id'])])

#agent = Agent(prompt_driver=HuggingFaceHubPromptDriver(repo_id="meta-llama/Llama-2-13b-chat-hf",api_token=os.environ['HUGGING_FACE_API_TOKEN']),
#              tools=[WebSearch(google_api_key=os.environ['google_api_key'], google_api_search_id=os.environ['google_api_search_id'])])


agent = Agent(prompt_driver=HuggingFaceHubPromptDriver(repo_id="upstage/Llama-2-70b-instruct",api_token=os.environ['HUGGING_FACE_API_TOKEN']),
              tools=[WebSearch(google_api_key=os.environ['google_api_key'], google_api_search_id=os.environ['google_api_search_id'])])

Chat(agent).start()