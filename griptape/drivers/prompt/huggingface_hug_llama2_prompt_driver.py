from attr import define, field, Factory
from transformers import LlamaTokenizerFast
from griptape.artifacts import TextArtifact
from griptape.utils import PromptStack
from griptape.drivers import HuggingFaceHubPromptDriver
from griptape.tokenizers import BaseTokenizer, HuggingFaceTokenizer

class HuggingfaceLlamaPromptDriver(HuggingFaceHubPromptDriver):
    
    def default_prompt_stack_to_string_converter(self, prompt_stack: PromptStack) -> str:
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
