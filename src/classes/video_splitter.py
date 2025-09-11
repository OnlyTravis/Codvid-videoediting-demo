from math import ceil
import cv2
from src.tools.logger import Logger
from src.tools.api_manager import APIManager
from src.tools.video_helper import VideoHelper
from src.tools.parser import ResponceParser
from src.classes.chunk import SmallChunk, Chunk
from src.classes.settings import VideoSplitSettings
from langchain_core.messages import HumanMessage, SystemMessage


class VideoSpliter:
    _frame_sequence_prompt: str
    _merge_chunk_prompt: str

    @classmethod
    def init(cls):
        # 1. Loading Prompt
        with open('./data/frame_sequence_prompt.txt', 'r') as f:
            cls._frame_sequence_prompt = f.read()
        with open('./data/merge_chunk_prompt.txt', 'r') as f:
            cls._merge_chunk_prompt = f.read()
    
    def __init__(self, 
                 video_path: str, 
                 video_description: str,
                 settings: VideoSplitSettings):
        self.video_path = video_path
        self.video_description = video_description
        self.settings = settings

    def split_video(self) -> list[Chunk]:
        # 1. Extract frames
        frame_count, frame_folder_path = VideoHelper.extract_all_frames(self.video_path, self.settings.seconds_per_frame, 768)

        # 2. Split into small chunks
        small_chunks = self._describe_small_chunks(frame_folder_path, frame_count)
        Logger.log_file('./small_chunks.txt', small_chunks)

        # 3. Merge into larger chunks


        # 4. (to be determined) 

    def _describe_small_chunks(self, frame_folder_path: str, frame_count: int) -> list[SmallChunk]:
        # 1. Calculate msg params
        msgs_count = frame_count // self.settings.frame_per_small_chunk
        msgs_list_count = ceil(msgs_count / self.settings.max_film_strip_per_message)
        msgs_per_list = msgs_count // msgs_list_count
        buffered_msgs_list = msgs_count - msgs_per_list * msgs_list_count # no. of lists with an additional message
        Logger.log_print(f'No. of Messages: {msgs_count}')
        Logger.log_print(f'No. of Message Lists: {msgs_list_count}')
        Logger.log_print(f'Messages per List: {msgs_per_list}')
        Logger.log_print(f'No. of Buffered Lists: {buffered_msgs_list}')

        # 2. Merge frames into frame sequence & create messages lists to send to LLM
        Logger.log_print('Merging frames & Building messages...')
        base_system_mes = SystemMessage(APIManager.format_prompt(
            self._frame_sequence_prompt,
            video_description=self.video_description, 
            frame_count=self.settings.frame_per_small_chunk
        ))
        Logger.create_folder('frame_sequences')
        msgs_lists = [[base_system_mes] for _ in range(msgs_list_count)]
        list_num = 0
        count = 0
        for i in range(0, msgs_count):
            # 2.1 Fetch frames and merge into frame sequence
            frames_arr = [cv2.imread(f'{frame_folder_path}/{x}.jpg') for x in range(3*i, 3*i+self.settings.frame_per_small_chunk)]
            frame_seq = VideoHelper.merge_frames_to_frame_seq(frames_arr)
            del frames_arr

            # 2.2 Log frame sequence output (if enabled)
            if (Logger.enabled):
                cv2.imwrite(Logger.to_path(f'frame_sequences/{i:0>3}.jpg'), frame_seq)

            # 2.3. Append Messages
            msgs_lists[list_num].append(
                HumanMessage(
                    content=[
                        {
                            "type": "text", 
                            "text": f"frame_sequence_{len(msgs_lists[list_num])}: "
                        },
                        {
                            "type": "image_url",
                            "image_url": VideoHelper.matlike_to_base64_url(frame_seq, 'jpeg'),
                        }
                    ]
                )
            )
            del frame_seq

            # 2.4 Check if next message list
            count += 1
            if (count >= msgs_per_list + 1 if (list_num+1 >= buffered_msgs_list) else 0):
                count = 0
                list_num += 1
        VideoHelper.remove_frame_folder(frame_folder_path)
        Logger.log_print("Finished Merging Frames & Building messages!")

        # 3. Ask LLM to describe frame sequence
        small_chunks: list[SmallChunk] = []
        index = 0
        for i in range(len(msgs_lists)):
            Logger.log_print(f'Sending Request {i+1} to llm...')
            ai_msg = APIManager.describe_frame_seq(messages=msgs_lists[i])
            Logger.log_print(f'Responce to Request {i+1} Received!')
            res = ai_msg.tool_calls[0]['args']
            Logger.log_file(f'extract_smlchunk_output_{i+1}.txt', res)
            small_chunks.extend([
                ResponceParser.parse_small_chunk_output(res['output'][j], self.settings, index+j) 
                for j in range(len(res['output']))
            ])
            index += len(res['output'])
            break # For testing
        return small_chunks
    
    def _merge_small_chunks(self, small_chunks: list[SmallChunk]) -> Chunk:
        pass