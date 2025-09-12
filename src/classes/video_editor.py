from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.tools.logger import Logger
from src.tools.api_manager import APIManager
from src.tools.prompts import PromptId, Prompts
from src.classes.chunk import Chunk

class VideoEditor:
    def __init__(self,
                 video_chunks: list[list[Chunk]],
                 user_prompt: str):
        self.video_chunks: list[list[Chunk]] = video_chunks
        self.removed_chunks: list[Chunk] = []
        self.user_prompt: str = user_prompt
        self.script: str = ''

    def auto_edit(self):
        if (self.script == ''):
            Logger.log_print('Video script not present, generating script...')
            self.generate_video_script()
        
        self.remove_redundant_chunks()
    
    def generate_video_script(self):
        '''
        Generates a new video script into self.script
        '''
        # 1. Creating Messages
        msgs: list[BaseMessage] = []
        msgs.append(SystemMessage(Prompts.get(PromptId.VIDEO_SCRIPT)))
        msgs.append(HumanMessage(self.user_prompt))
        for i in range(len(self.video_chunks)):
            text = f'Video {i}:\n'
            text += '\n'.join([
                f"Chunk {j}'s Summary: {self.video_chunks[i][j].summary}"
                for j in range(len(self.video_chunks[i]))
            ])
            msgs.append(HumanMessage(text))
        
        # 2. Send to llm
        script = APIManager.generate_video_script(msgs)
        Logger.log_file('./video_script', script)
        self.script = script
    
    def remove_redundant_chunks(self):
        '''
        Detect redundant chunk based on chunk's description & self.script
        Moves redundant chunks into self.removed_chunks
        '''
        # 1. Assemble Messages
        msgs: list[BaseMessage] = []
        msgs.append(SystemMessage(Prompts.get(PromptId.REMOVE_CHUNKS)))
        msgs.append(HumanMessage(f'Video Script: \n{self.script}'))
        for i in range(len(self.video_chunks)):
            text = ''
            for j in range(len(self.video_chunks[i])):
                chunk = self.video_chunks[i][j]
                text += f'Video {i+1}, Chunk_id({j+1}): {chunk.summary}\n'
            msgs.append(HumanMessage(text))

        # 2. Send to llm
        APIManager.detect_redundant_chunks(msgs)

    