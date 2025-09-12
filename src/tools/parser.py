from src.classes.chunk import Chunk, SmallChunk
from src.classes.settings import VideoSplitSettings

class ResponceParser:
    @classmethod
    def parse_small_chunk_output(cls, chunk_data: dict, split_settings: VideoSplitSettings, msg_index: int) -> SmallChunk:
        '''
        Converts structure dict output from LLM to SmallChunk\n
        msg_index: Starts from 0
        '''
        time_range = split_settings.get_chunk_time_range(msg_index)
        return SmallChunk(msg_index, chunk_data['description'], time_range[0], time_range[1])
    
    @classmethod
    def parse_chunk_output(cls, chunk_data: dict, split_settings: VideoSplitSettings, base_index: int) -> Chunk:
        '''
        Converts structure dict output from LLM to Chunk\n
        base_index: message index of the first message in the message list
        '''
        start_index: int = chunk_data['start']+base_index-1
        end_index: int = chunk_data['end']+base_index-1
        return Chunk(id=-1, 
                     summary=chunk_data['summary'], 
                     chunk_start=start_index, 
                     chunk_end=end_index,
                     time_start=split_settings.get_chunk_time_range(start_index)[0],
                     time_end=split_settings.get_chunk_time_range(end_index)[1])