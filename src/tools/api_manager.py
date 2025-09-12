import getpass
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage

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
        cls._google_llm_frame_seq = cls._google_llm.bind_tools([SmallChunksOutputSchema], tool_choice='SmallChunksOutputSchema')
        cls._google_llm_group_sml_chunk = cls._google_llm.bind_tools([GroupChunksOutputSchema], tool_choice='GroupChunksOutputSchema')
    
    @classmethod
    def describe_frame_seq(cls, messages_lists: list[list[BaseMessage]], split_settings: VideoSplitSettings) -> list[SmallChunk]:
        small_chunks: list[SmallChunk] = []
        base_index = 0
        for i in range(len(messages_lists)):
            # 1. Sending Request
            Logger.log_print(f'Sending Description Frame Sequences Request {i+1} to llm...')
            ai_msg = cls._google_llm_frame_seq.invoke(messages_lists[i])
            Logger.log_print(f'Responce to Description Frame Sequences Request {i+1} Received!')

            # 2. Processing Responce
            Logger.log_file(f'describe_frame_seq_raw_output_{i+1}.txt', ai_msg)
            res = ai_msg.tool_calls[0]['args']
            small_chunks.extend([
                ResponceParser.parse_small_chunk_output(res['output'][j], split_settings, base_index+j) 
                for j in range(len(res['output']))
            ])
            base_index += len(res['output'])
        Logger.log_file(f'small_chunks.txt', list(map(lambda x: x.__dict__, small_chunks)))
        return small_chunks

    @classmethod
    def group_small_chunks(cls, messages_lists: list[list[BaseMessage]], split_settings: VideoSplitSettings) -> list[Chunk]:
        chunks: list[Chunk] = []
        base_index = 0
        previous_chunk_data: dict | None = None
        last_base_index = 0
        for i in range(len(messages_lists)):
            # 1. Sending Request
            Logger.log_print(f'Sending Group Small Chunks Request {i+1} to llm...')
            ai_msg = cls._google_llm_group_sml_chunk.invoke(messages_lists[i])
            Logger.log_print(f'Responce to Group Small Chunks Request {i+1} Received!')

            # 2. Merge first with previous last (if any)
            Logger.log_file(f'group_chunk_raw_output_{i+1}.txt', ai_msg)
            res = ai_msg.tool_calls[0]['args']
            if (previous_chunk_data == None):
                chunks.append(ResponceParser.parse_chunk_output(res['output'][0], split_settings, base_index))
            else:
                previous_chunk_data['end'] += res['output'][0]['end'] - 1
                previous_chunk_data['summary'] += f" {res['output'][0]['summary']}"
                chunks.append(ResponceParser.parse_chunk_output(previous_chunk_data, split_settings, last_base_index))

            # 3. Process responces
            chunks.extend([
                ResponceParser.parse_chunk_output(res['output'][j], split_settings, base_index) 
                for j in range(1, len(res['output'])-1)
            ])

            # 4. Save last for next (if any)
            if (i+1 == len(messages_lists)):
                chunks.append(ResponceParser.parse_chunk_output(res['output'][-1], split_settings, base_index))
                break
            previous_chunk_data = res['output'][-1]
            last_base_index = base_index
            base_index += res['output'][-1]['end']-1 # -1 for padding
        Logger.log_file(f'chunks.txt', list(map(lambda x: x.__dict__, chunks)))
        return chunks
    
    @classmethod
    def generate_video_script(cls, msgs: list[BaseMessage]) -> str:
        # 1. Send to LLM
        ai_msg = cls._google_llm.invoke(msgs)
        Logger.log_file('video_script_raw_output.txt', ai_msg)
        return ai_msg.content
    
    @classmethod 
    def detect_redundant_chunks(cls, msgs: list[BaseMessage]) -> list[int]:
        # 1. Send to LLM
        Logger.log_file('detect_redundant_chunks_input.txt', msgs)
        ai_msg = cls._google_llm.invoke(msgs)
        Logger.log_file('detect_redundant_chunks_raw_output.txt', ai_msg)

        # 2. Parse output