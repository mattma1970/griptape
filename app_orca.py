'''
Sample app to test prompt driver for a locally installed model where generation is directly invoked rather than via an API.
Because the model calls don't conform to a standard protocol bespoke interfaces used by the driver to call the chat/text generation functions on the model have been created. 
The idea was to explore building a bespoke interface and integrate that into Griptape. This allows models to be used directly from their repos: an advantage when you want to 
hack the models or go beyond what it possible with common API providers. 
'''

from dotenv import load_dotenv

# Griptape Items
from griptape.structures import Agent
from griptape.utils import Chat #   <-- Added Chat
from griptape.drivers import LocalOpenOrcaPromptDriver
import logging
import argparse
import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import optimum
from generation import LocalLlama
from types import SimpleNamespace
from griptape.tools import WebSearch


from models.llama.llama import Llama  #update this to point to the folder where you download the code. 

# Create the agent
def example(model,tokenizer, args):   
    params =  {
                "max_new_tokens": args.max_gen_len, #new tokens per generation
                "max_tokens": args.max_seq_len, # maximum context window+new_tokens.
                "temperature":args.temperature
            }
    agent = Agent(
                    logger_level=logging.INFO, 
                    prompt_driver=LocalOpenOrcaPromptDriver(
                                                            inference_resource=model,
                                                            tokenizer=tokenizer,
                                                            task='chat',
                                                            params = params,
                                                            use_gpu=True,
                                                            ),
                    tools=[WebSearch(google_api_key=os.environ['google_api_key'], google_api_search_id=os.environ['google_api_search_id'])]
                )
    # Begin Chatting
    Chat(agent).start()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',type=str, default='upstream-griptape/models/')
    parser.add_argument('--model_path', type=str, default='Mistral-7B-OpenOrca', help='Relative path to root_path')
    parser.add_argument('--model_name', type=str, default = 'Mistral-7B-OpenOrca')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.4)
    parser.add_argument('--max_seq_len',type=int, default=4095)
    parser.add_argument('--max_gen_len',type=int, default=1024)
    parser.add_argument('--max_batch_size',type=int, default=1)
    parser.add_argument('--debug',action='store_true', default =False, help='Be far more chatty about the internals.')
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'  # Replace with the appropriate address
    os.environ['MASTER_PORT'] = '1234'  # Replace with the appropriate port
    os.environ['RANK'] = '0'  # Replace with the rank of the current process
    os.environ['WORLD_SIZE'] = '1'  # Replace with the total number of processes


    #llm=build_llm(args)
    assets_path = os.path.join(args.root_path,args.model_name)
    llm = AutoModelForCausalLM.from_pretrained(assets_path, device_map="auto", use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(assets_path,)
    example(llm,tokenizer, args)