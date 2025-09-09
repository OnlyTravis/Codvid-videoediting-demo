import cv2
from src.classes.logger import Logger
from src.classes.api_manager import APIManager
from src.classes.chunk import Chunk
from src.classes.video_manager import VideoManager
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

_prompt: str
with open('./data/chunk_prompt.txt', 'r') as f:
    _prompt = f.read()

class ExtractChunkSettings:
    small_chunk_per_second: float
    frame_per_small_chunk: int

    def __init__(self,
                 small_chunk_per_second: float,
                 frame_per_small_chunk: int):
        self.small_chunk_per_second = small_chunk_per_second
        if (frame_per_small_chunk % 2 != 1):
            raise Exception("frame_per_small_chunk must be an odd number in ExtractChunkSettings!")
        self.frame_per_small_chunk = frame_per_small_chunk

    @property
    def frames_per_second(self):
        return self.small_chunk_per_second * self.frame_per_small_chunk
    @property
    def seconds_per_frame(self):
        return 1 / (self.small_chunk_per_second * self.frame_per_small_chunk)
    
    @classmethod
    def defaultSettings(cls):
        return cls(1, 3)

def extract_chunks(video_path: str, user_prompt: str, extract_settings: ExtractChunkSettings) -> list[Chunk]:
    # 1. Extract frames
    frame_count, frame_folder_path = VideoManager.extract_all_frames(video_path, extract_settings.seconds_per_frame, 1024)

    # 2. Merge into film strip & create messages
    messages: list[BaseMessage] = [SystemMessage(_prompt.format(
        prompt=user_prompt, 
        frame_count=extract_settings.frame_per_small_chunk
    ))]
    for i in range(0, frame_count-extract_settings.frame_per_small_chunk+1, 3):
        # 2.1 Fetch frames and merge into film strip
        frames_arr = [cv2.imread(f'{frame_folder_path}/{x}.jpg') for x in range(i, i+3)]
        film_strip = VideoManager.merge_frames_to_file_strip(frames_arr)
        del frames_arr

        # 2.2 Log film strip output (if enabled)
        if (Logger.enabled):
            cv2.imwrite(Logger.to_path(f'film_strip_{i//3:0>3}.jpg'), film_strip)

        # 2.3. Append Messages
        messages.append(
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": VideoManager.matlike_to_base64_url(film_strip, 'jpeg'),
                    }
                ]
            )
        )
        del film_strip
    VideoManager.remove_frame_folder(frame_folder_path)
    Logger.log_print("Finished Merging Frames")
    
    # 3. Send to LLM
    APIManager.describe_film_strips(messages=messages)

    return []