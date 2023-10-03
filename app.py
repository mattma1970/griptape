from dotenv import load_dotenv

# Griptape Items
from griptape.structures import Agent
from griptape.utils import Chat #   <-- Added Chat
from griptape.drivers import LocalLlamaPromptDriver
import logging
import asyncio
import aiohttp


# Load environment variables
load_dotenv()

tokenizer_path = "/home/mtman/Documents/Repos/llama/tokenizer.model"
# Create the agent
def example():
    agent = Agent(logger_level=logging.ERROR, prompt_driver=LocalLlamaPromptDriver(inference_endpoint='http://localhost:8080', task='chat', tokenizer_path=tokenizer_path, async_session=None))
    # Begin Chatting
    Chat(agent).start()

example()