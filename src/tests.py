from src.classes.logger import Logger
from src.workflow import workflow

def test_1():
    '''
    Test 1: Room tour
    Difficulty: Easy
    Task: Remove parts of video according to user's input.
    Raw_footage: 1
    '''
    Logger.log_print("Test 1 Started!")
    file_name = './_test_data/test_1.mp4'
    prompt = "This is a video for a tour of my house, please cut out the part where it shows my toilet and when I'm not moving for too long"

    workflow(file_name, prompt)