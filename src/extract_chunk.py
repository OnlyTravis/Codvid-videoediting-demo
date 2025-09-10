from math import ceil
import cv2
from src.tools.logger import Logger
from src.tools.api_manager import APIManager
from src.tools.video_manager import VideoManager
from src.classes.chunk import Chunk
from src.classes.settings import ExtractChunkSettings
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

_prompt: str
with open('./data/chunk_prompt.txt', 'r') as f:
    _prompt = f.read()

def extract_chunks(video_path: str, user_prompt: str, extract_settings: ExtractChunkSettings) -> list[Chunk]:
    # 1. Extract frames
    frame_count, frame_folder_path = VideoManager.extract_all_frames(video_path, extract_settings.seconds_per_frame, 768)

    # 2. Calculate split msg lists
    msgs_count = frame_count // extract_settings.frame_per_small_chunk
    msgs_list_count = ceil(msgs_count / extract_settings.max_film_strip_per_message)
    msgs_per_list = msgs_count // msgs_list_count
    buffered_msgs_list = msgs_count - msgs_per_list * msgs_list_count # no. of lists with an additional message
    Logger.log_print(f'No. of Messages: {msgs_count}')
    Logger.log_print(f'No. of Message Lists: {msgs_list_count}')
    Logger.log_print(f'Messages per List: {msgs_per_list}')
    Logger.log_print(f'No. of Buffered Lists: {buffered_msgs_list}')

    # 3. Merge into film strip & create messages lists
    Logger.log_print('Merging frames & Building messages...')
    base_system_mes = SystemMessage(_prompt.format(
        prompt=user_prompt, 
        frame_count=extract_settings.frame_per_small_chunk
    ))
    msgs_lists = [[base_system_mes] for _ in range(msgs_list_count)]
    list_num = 0
    count = 0
    for i in range(0, msgs_count):
        # 2.1 Fetch frames and merge into film strip
        frames_arr = [cv2.imread(f'{frame_folder_path}/{x}.jpg') for x in range(3*i, 3*i+extract_settings.frame_per_small_chunk)]
        film_strip = VideoManager.merge_frames_to_file_strip(frames_arr)
        del frames_arr

        # 2.2 Log film strip output (if enabled)
        if (Logger.enabled):
            cv2.imwrite(Logger.to_path(f'film_strip_{i:0>3}.jpg'), film_strip)

        # 2.3. Append Messages
        msgs_lists[list_num].append(
            HumanMessage(
                content=[
                    {
                        "type": "text", 
                        "text": f"Film_strip_{len(msgs_lists[list_num])}: "
                    },
                    {
                        "type": "image_url",
                        "image_url": VideoManager.matlike_to_base64_url(film_strip, 'jpeg'),
                    }
                ]
            )
        )
        del film_strip

        # 2.4 Check if increment list_num
        count += 1
        if (count >= msgs_per_list + 1 if (list_num+1 >= buffered_msgs_list) else 0):
            count = 0
            list_num += 1

    VideoManager.remove_frame_folder(frame_folder_path)
    Logger.log_print("Finished Merging Frames & Building messages!")
    
    # 3. Send to LLM
    for i in range(len(msgs_lists)):
        res = APIManager.describe_film_strips(messages=msgs_lists[i])
        Logger.log_file(f'extract_smlchunk_output_{i+1}.json', res)

    return []