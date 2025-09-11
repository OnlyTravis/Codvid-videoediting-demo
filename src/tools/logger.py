import json
import os
from typing import IO, Any

class LoggerSettings:
    def __init__(self, log_index: int):
        self.log_index: int = log_index
    
    def toDict(self) -> dict:
        return {
            'log_index': self.log_index,
        }

    @classmethod
    def fromDict(cls, d: dict) -> "LoggerSettings":
        return cls(d['log_index'])
    
    @classmethod
    def defaultSettings(cls) -> "LoggerSettings":
        return cls(0)

class Logger:
    enabled: bool
    log_folder_path: str
    _settings: LoggerSettings

    @classmethod
    def init(cls, enabled: bool):
        # 1. Check if log is enabled
        cls.enabled = enabled
        if (not enabled): return

        # 2. Get log settings & increment log index
        try:
            with open('./data/log_settings.json', 'r') as f:
                cls._settings = LoggerSettings.fromDict(json.loads(f.read()))
        except FileNotFoundError | json.JSONDecodeError:
            cls._settings = LoggerSettings.defaultSettings()
        cls._settings.log_index += 1
        with open('./data/log_settings.json', 'w') as f:
            f.write(json.dumps(cls._settings.toDict()))

        # 3. Create log folder
        if (not os.path.isdir('./logs')):
            os.mkdir('./logs')
        cls.log_folder_path = f'./logs/log_{cls._settings.log_index:0>2}'
        os.mkdir(cls.log_folder_path)

        # 4. Create log file
        with open(cls.to_path('log.txt'), 'w') as f:
            f.write(f'***********Start of Log {cls._settings.log_index}***********\n')

    @classmethod
    def to_path(cls, file_name: str):
        return f'{cls.log_folder_path}/{file_name}'
    
    @classmethod
    def write_to_log(cls, string: str):
        if (not cls.enabled): return
        with open(cls.to_path('log.txt'), 'a') as f:
            f.write(string+'\n')

    @classmethod
    def log_print(cls, data: Any):
        if (not cls.enabled): return
        cls.write_to_log(str(data))
        print(data)

    @classmethod
    def log_file(cls, file_name: str, data: str | bytes | object):
        if (not cls.enabled): return
        f: IO
        if isinstance(data, str):
            f = open(cls.to_path(file_name), 'w')
        elif isinstance(data, bytes):
            f = open(cls.to_path(file_name), 'wb')
        else:
            data = str(data)
            f = open(cls.to_path(file_name), 'w')
        f.write(data)
        f.close()
    
    @classmethod
    def create_folder(cls, folder_name: str):
        if (not cls.enabled): return
        os.mkdir(cls.to_path(folder_name))