from src.classes.chunk import SmallChunk
from src.classes.settings import VideoSplitSettings

class ResponceParser:
    @classmethod
    def parse_small_chunk_output(cls, chunk_data: dict, split_settings: VideoSplitSettings, msg_index: int) -> SmallChunk:
        '''
        Converts structure dict output from LLM to SmallChunk
        msg_index: Starts from 0
        '''
        time_range = split_settings.get_chunk_time_range(msg_index)
        return SmallChunk(msg_index, chunk_data['description'], time_range[0], time_range[1])