import getpass
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

from src.classes.logger import Logger

film_strip_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "film_strip_name": {
                "type": "string",
                "description": "The name should be 'filmstrip_<index>' where index starts at 1."
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the film strip."
            }
        },
        "required": ["description"]
    }
}

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
        cls._google_llm_film_strip = cls._google_llm.with_structured_output(schema=film_strip_schema, method="json_mode")
    
    @classmethod
    def describe_film_strips(cls, messages: list[BaseMessage]):
        Logger.log_file('film_strip_input.txt', messages)
        res = cls._google_llm_film_strip.invoke(messages)
        Logger.log_file('film_strip_output.json', res)

