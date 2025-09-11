import dotenv
from src.tools.logger import Logger
from src.tools.api_manager import APIManager
from src.classes.video_splitter import VideoSpliter

from src.tests import test_1

def main():
    # 1. Initialization
    debug = True
    dotenv.load_dotenv()

    Logger.init(enabled=True)
    APIManager.init()
    VideoSpliter.init()

    test_1()

main()