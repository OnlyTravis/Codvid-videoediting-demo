from src.classes.settings import VideoSplitSettings
from src.classes.video_splitter import VideoSpliter
from src.classes.video_editor import VideoEditor
from src.tools.logger import Logger

def test_1():
    '''
    Test 1: Room tour
    Difficulty: Easy
    Task: Remove parts of video according to user's input.
    Description: A simple house tour video with a standing still part and a black screen part
    Raw_footage: 1
    '''
    Logger.log_print("Test 1 Started!")
    file_name = './_test_data/test_1.mp4'
    video_descriptions = ['A tour for my house']
    prompt = "This is a video for a tour of my house, please cut out the part where it shows my toilet and when I'm not moving for too long"

    split_settings = VideoSplitSettings(
        small_chunk_per_second=0.67,
        frame_per_small_chunk=3,
        max_frame_seq_per_req=17,
        max_small_chunk_per_req=30,
    )
    video_splitter = VideoSpliter(file_name, video_descriptions[0], split_settings)
    chunks = video_splitter.split_video()
    editor = VideoEditor([chunks], prompt)
    editor.auto_edit()
