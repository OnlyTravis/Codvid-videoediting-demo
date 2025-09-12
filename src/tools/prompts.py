from enum import Enum

class PromptId(Enum):
    MERGE_CHUNKS = 0
    REMOVE_CHUNKS = 1
    VIDEO_SCRIPT = 2
    DESCRIBE_FRAME_SEQUENCE = 3

_file_names = {
    PromptId.MERGE_CHUNKS: 'merge_chunk_prompt.txt',
    PromptId.REMOVE_CHUNKS: 'remove_chunk_prompt.txt',
    PromptId.VIDEO_SCRIPT: 'video_script_prompt.txt',
    PromptId.DESCRIBE_FRAME_SEQUENCE: 'frame_sequence_prompt.txt'
}

class Prompts:
    _prompts: dict[int, str] = {}

    @classmethod
    def init(cls):
        # 1. Load prompts
        for prompt_id, file_name in _file_names.values():
            with open(f'./data/{file_name}') as f:
                cls._prompts[prompt_id] = f.read()
    
    @classmethod
    def get(cls, prompt_id: PromptId) -> str:
        '''
        Fetch a prompt from prompt_id.
        '''
        return cls._prompts[prompt_id]

    @classmethod
    def get_formatted(cls, prompt_id: PromptId, *args, **kwargs) -> str:
        '''
        Fetch and formats a prompt from prompt_id.
        '''
        return cls.format_prompt(cls.get(prompt_id), *args, **kwargs)

    @classmethod
    def format_prompt(cls, prompt: str, *args, **kwargs) -> str:
        '''
        Formats prompt in '{%<name>%}' format to prevent conflict with curly brackets\n
        e.g. 'xx{%id:0>2%}xxx' = f'xx{id:0>2}xxx'
        '''
        return prompt.replace('{', '{{').replace('}', '}}').replace('{{%', '{').replace('%}}', '}').format(*args, **kwargs)