import getpass
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

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
        cls._google_llm_film_strip = cls._google_llm.bind_tools([SmallChunksOutputSchema])
    
    @classmethod
    def describe_film_strips(cls, messages: list[BaseMessage]) -> BaseMessage:
        res = cls._google_llm_film_strip.invoke(messages)
        return res
