from vllm import LLM, SamplingParams
from dotenv import load_dotenv
import openai
import pprint
import requests
from griptape.structures import Agent

from griptape.tokenizers import HuggingFaceTokenizer
from griptape.tools import WebSearch
from griptape.utils import Chat, PromptStack
from transformers import AutoTokenizer
from huggingface_hub import InferenceApi
import os

#load_dotenv()
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"
#model_list= requests.get('http://localhost:8000/v1/models')
#print(model_list.content)


from griptape.drivers import OpenAiChatPromptDriver, HuggingFaceHubPromptDriver, HuggingFacevLLMPromptDriver
#
#How to invoke the model using vllm
#params = SamplingParams(max_tokens=512)
#llm = LLM('Open-Orca/Mistral-7B-OpenOrca')

prompt = """
This is a cafe menu: \nAVO BRUSCHETTA ~ 22\nOn sourdough, heirloom tomatoes, fetta,\nsecret salt, basil & walnut pesto.\nadd poached egg 3\n\nSOUP OF THE DAY - 16\nWith rustic bread.\n\nPlease Sow mW capiorel for\n\nhelio.\nof gourrel Sundiicpes and hore\nmeade. deseer le\n\nWWW.TWISTESPRESSO.COM\n\nPlease order at the counter = 10% surcharge Sundays & 15% Public ITolidays\n\n \n\x0c' Please convert the menu to a valid JSON string that follows the schema {"dish":<<name>>, "price":<<price>>, "ingredients_list":[ingredint1, ingredient2,...],"description":<<summary of dish in fewer than 5 words>>}
"""

""" outputs = llm.generate(prompt, params)
generated_text = outputs[0].outputs[0].text
print (generated_text) """

prompt2 = """ This is a cafe menu 
'rp 7 I Ss | y\nIW\nESPRESSO & WINE\n\nWINTER BRUNCH MENU AVAILABLE UNTIL 1.30PM\n\n \n\n \n\nRUSTIC TOAST - 8.5\nSourdough or Ciabatta.\nFig & lemon jam, orange marmalade, strawberry\njam, peanut butter or vegemite.\n\nHOT CHEESE - 9.5\n\nWith tomato & oregano on sourdough.\n\nFRUIT TOAST - 8.5\nApricots, dates, fig & nuts, toasted with butter.\n\nGRANOLA CUP GF - 10.5\nHouse-made granola, greek yoghurt, strawberry\njam and strawberries.\n\nACAI BOWL GF - 17.5\nHouse-made granola, banana, strawberry.\nadd peanut butter 2\n\nITALIAN OMELETTE - 23\nTomato, mozarella, topped with basil & walnut\npesto, mixed leaves, ciabatta.\n\nMEXICAN EGGS - 23\nMixed beans cooked with chilli, garlic, onion and\nherbs in rich tomato sauce, baked with\nmozzerella, two poached eggs, ciabatta.\n\nEGGS BENEDICT - 26\nTwo poached eggs, spinach, and hollandaise sauce\nserved on sourdough.\nChoice of : salmon, bacon or ham.\n\nEGGS POACHED OR FRIED - 13\n\nServed on sourdough or ciabatta.\n\nEXTRAS:\n\nSalmon, Gourmet Pork Sausages 6\nBacon, Haloumi 5\nAvocado, Grilled Tomato 4\nFree-Range Egg 3\nGluten Free Bread 2\n\n \n\nTWIST BIG BREAKFAST ~ 28\nTwo poached or fried eggs, gourmet pork\nsausages, haloumi, grilled tomato, spinach,\nsourdough,\n\nBREKKY ROLL ~ 13\nBacon & egy on a milk bun.\nSauce: tomato, BBQ, chipotle mayo or aioli.\nadd cheese 1 smashed avo 2\n\nTWIST WINTER BUN ~ 17\nBacon or haloumi, egg, tomato, caramelised\nonion, mixed leaves, melted cheese, aioli.\nadd smashed avo 2\n\nAVO BRUSCHETTA ~ 22\nOn sourdough, heirloom tomatoes, fetta,\nsecret salt, basil & walnut pesto.\nadd poached egg 3\n\nSOUP OF THE DAY - 16\nWith rustic bread. \n\nPlease Sow mW capiorel for\n\nhelio.\nof gourrel Sundiicpes and hore\nmeade. deseer le\n\[nWWW.TWISTESPRESSO.COM](http://nwww.twistespresso.com/)\n\nPlease order at the counter = 10% surcharge Sundays & 15% Public I Tolidays\n\n \n\x0c' Show me all the dishes that have eggs as a main ingredient and include the price?
"""

#tokenizer = HuggingFaceTokenizer(tokenizer=AutoTokenizer.from_pretrained('Open-Orca/Mistral-7B-OpenOrca'))

agent = Agent(prompt_driver = HuggingFacevLLMPromptDriver(
                model='/home/mtman/Documents/Repos/griptape_experiments/upstream-griptape/models/Mistral-7B-OpenOrca',
                api_base = 'http://localhost:8000/',
                max_tokens=4000,
                temperature=1.1,
                params={"stop_token_ids":[2,32000],"max_tokens":4000} #?? Sampling parameters as enumerated in SampleParameters()
                ), 
                #tools = [WebSearch(google_api_key=os.environ['google_api_key'], google_api_search_id=os.environ['google_api_search_id'])],

        )

Chat(agent).start()

#completion = openai.Completion.create(model="/home/mtman/Documents/Repos/griptape_experiments/upstream-griptape/models/Mistral-7B-OpenOrca/",prompt="San Francisco is a")
#completion = openai.Completion.create(model='/home/mtman/Documents/Repos/griptape_experiments/upstream-griptape/models/Mistral-7B-OpenOrca',prompt="San Francisco is a",max_tokens="256")
#completion = openai.Completion.create(model='/home/mtman/Documents/Repos/griptape_experiments/upstream-griptape/models/Mistral-7B-OpenOrca',prompt=prompt2,max_tokens=1024, temperature=0.95)

#print(completion)


#outputs = llm.generate(prompt2, params)
#generated_text = outputs[0].outputs[0].text
#print (generated_text)