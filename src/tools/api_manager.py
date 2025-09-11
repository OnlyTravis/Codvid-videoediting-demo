import getpass
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, AIMessage

from src.classes.structured_output import SmallChunksOutputSchema
from src.tools.logger import Logger

class APIManager:
    _google_llm: ChatGoogleGenerativeAI
    _google_llm_film_strip: ChatGoogleGenerativeAI

    @classmethod
    def init(cls):
        # 1. Get API Key
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
        
        # 2. Initialize Google LLM
        cls._google_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        cls._google_llm_frame_seq = cls._google_llm.bind_tools([SmallChunksOutputSchema])
    
    @classmethod
    def describe_frame_seq(cls, messages: list[BaseMessage]) -> AIMessage:
        res = cls._google_llm_frame_seq.invoke(messages)
        return res
    
    @classmethod
    def format_prompt(cls, prompt: str, *args, **kwargs) -> str:
        '''
        Formats prompt in '{%<name>%}' format to prevent conflict with curly brackets\n
        e.g. 'xx{%id:0>2%}xxx' = f'xx{id:0>2}xxx'
        '''
        return prompt.replace('{', '{{').replace('}', '}}').replace('{{%', '{').replace('%}}', '}').format(*args, **kwargs)
