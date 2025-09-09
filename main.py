import dotenv
from src.classes.logger import Logger
from src.classes.api_manager import APIManager
from src.workflow import workflow

from src.tests import test_1

def main():
    # 1. Initialization
    debug = True
    dotenv.load_dotenv()

    Logger.init(enabled=True)
    APIManager.init()

    test_1()

main()