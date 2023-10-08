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
from griptape.drivers import LocalLlamaPromptDriver
import logging
import argparse
import os

from models.llama.llama import Llama  #update this to point to the folder where you download the code. 

# Create the agent
def example(model,args):
    #agent = Agent(logger_level=logging.ERROR, prompt_driver=LocalLlamaPromptDriver(inference_endpoint='http://localhost:8080', task='chat', tokenizer_path=tokenizer_path))       
    params =  {
                "max_new_tokens": args.max_gen_len, #new tokens per generation
                "max_tokens": args.max_seq_len, # maximum context window+new_tokens.
            }
    agent = Agent(logger_level=logging.ERROR, prompt_driver=LocalLlamaPromptDriver(inference_resource=model, task='chat', tokenizer_path=args.tokenizer_path, params = params))
    # Begin Chatting
    Chat(agent).start()


def build_llm(args):
    llm = Llama.build(
        ckpt_dir= os.path.join(args.root_path,args.model_path),
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

    return llm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',type=str, default='/home/mtman/Documents/Repos/griptape_experiments/upstream-griptape/models/llama/')
    parser.add_argument('--model_path', type=str, default='llama-2-7b-chat', help='Relative path to model checkpoint')
    parser.add_argument('--tokenizer_path', type=str, default='/home/mtman/Documents/Repos/llama/tokenizer.model')
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.4)
    parser.add_argument('--max_seq_len',type=int, default=2000)
    parser.add_argument('--max_gen_len',type=int, default=512)
    parser.add_argument('--max_batch_size',type=int, default=4)
    parser.add_argument('--debug',action='store_true', default =False, help='Be far more chatty about the internals.')
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'  # Replace with the appropriate address
    os.environ['MASTER_PORT'] = '1234'  # Replace with the appropriate port
    os.environ['RANK'] = '0'  # Replace with the rank of the current process
    os.environ['WORLD_SIZE'] = '1'  # Replace with the total number of processes

    llm=build_llm(args)

    example(llm,args)