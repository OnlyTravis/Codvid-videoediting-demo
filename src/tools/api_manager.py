import getpass
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, AIMessage

from src.classes.chunk import Chunk, SmallChunk
from src.classes.settings import VideoSplitSettings
from src.classes.structured_output import GroupChunksOutputSchema, SmallChunksOutputSchema
from src.tools.logger import Logger
from src.tools.parser import ResponceParser

class APIManager:
    _google_llm: ChatGoogleGenerativeAI
    _google_llm_frame_seq: ChatGoogleGenerativeAI
    _google_llm_group_sml_chunk: ChatGoogleGenerativeAI

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
        cls._google_llm_group_sml_chunk = cls._google_llm.bind_tools([GroupChunksOutputSchema])
    
    @classmethod
    def describe_frame_seq(cls, messages_lists: list[list[BaseMessage]], split_settings: VideoSplitSettings) -> list[SmallChunk]:
        small_chunks: list[SmallChunk] = []
        index = 0
        for i in range(len(messages_lists)):
            # 1. Sending Request
            Logger.log_print(f'Sending Description Frame Sequences Request {i+1} to llm...')
            ai_msg = cls._google_llm_frame_seq.invoke(messages_lists[i])
            Logger.log_print(f'Responce to Description Frame Sequences Request {i+1} Received!')

            # 2. Processing Responce
            Logger.log_file(f'describe_frame_seq_raw_output_{i+1}.txt', ai_msg)
            res = ai_msg.tool_calls[0]['args']
            small_chunks.extend([
                ResponceParser.parse_small_chunk_output(res['output'][j], split_settings, index+j) 
                for j in range(len(res['output']))
            ])
            index += len(res['output'])
        return small_chunks

    @classmethod
    def group_small_chunks(cls, messages_lists: list[list[BaseMessage]], split_settings: VideoSplitSettings) -> list[Chunk]:
        chunks: list[Chunk] = []
        index = 0
        for i in range(len(messages_lists)):
            # 1. Sending Request
            Logger.log_print(f'Sending Group Small Chunks Request {i+1} to llm...')
            ai_msg = cls._google_llm_group_sml_chunk.invoke(messages_lists[i])
            Logger.log_print(f'Responce to Group Small Chunks Request {i+1} Received!')

            # 2. Processing Responce
            res = ai_msg.tool_calls[0]['args']
            Logger.log_file(f'group_chunk_output_{i+1}.txt', res)
            chunks.extend([
                ResponceParser.parse_chunk_output(res['output'][j], split_settings, index+j) 
                for j in range(len(res['output']))
            ])
            index += len(res['output'])
        return chunks
    
    @classmethod
    def format_prompt(cls, prompt: str, *args, **kwargs) -> str:
        '''
        Formats prompt in '{%<name>%}' format to prevent conflict with curly brackets\n
        e.g. 'xx{%id:0>2%}xxx' = f'xx{id:0>2}xxx'
        '''
        return prompt.replace('{', '{{').replace('}', '}}').replace('{{%', '{').replace('%}}', '}').format(*args, **kwargs)
