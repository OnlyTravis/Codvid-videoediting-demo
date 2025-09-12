from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.tools.logger import Logger
from src.tools.api_manager import APIManager
from src.classes.chunk import Chunk

class VideoEditor:
    _generate_video_script_prompt: str

    @classmethod
    def init(cls):
        with open('./data/video_script_prompt.txt') as f:
            cls._generate_video_script_prompt = f.read()

    def __init__(self,
                 video_chunks: list[list[Chunk]],
                 user_prompt: str):
        self.video_chunks: list[list[Chunk]] = video_chunks
        self.user_prompt: str = user_prompt

    def auto_edit(self):
        script = self.generate_video_script()
    
    def generate_video_script(self) -> str:
        # 1. Creating Messages
        msgs: list[BaseMessage] = []
        msgs.append(SystemMessage(self._generate_video_script_prompt))
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
        return script