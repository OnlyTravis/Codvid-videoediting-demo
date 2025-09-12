import dotenv
from src.tools.logger import Logger
from src.tools.prompts import Prompts
from src.tools.api_manager import APIManager
from src.classes.video_splitter import VideoSpliter
from src.classes.video_editor import VideoEditor

from src.tests import test_1

def main():
    # 1. Initialization
    debug = True
    dotenv.load_dotenv()

    Logger.init(enabled=debug)
    Prompts.init()
    APIManager.init()
    VideoSpliter.init()

    # 2. Run Test
    test_1()

main()