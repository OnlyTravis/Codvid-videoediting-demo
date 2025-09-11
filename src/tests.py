from src.classes.settings import VideoSplitSettings
from src.classes.video_splitter import VideoSpliter
from src.tools.logger import Logger

def test_1():
    '''
    Test 1: Room tour
    Difficulty: Easy
    Task: Remove parts of video according to user's input.
    Raw_footage: 1
    '''
    Logger.log_print("Test 1 Started!")
    file_name = './_test_data/test_1.mp4'
    video_descriptions = ['A tour for my house']
    prompt = "This is a video for a tour of my house, please cut out the part where it shows my toilet and when I'm not moving for too long"

    split_settings = VideoSplitSettings(
        small_chunk_per_second=0.67,
        frame_per_small_chunk=3,
        max_film_strip_per_message=30
    )
    video_splitter = VideoSpliter(file_name, video_descriptions[0], split_settings)
    video_splitter.split_video()
