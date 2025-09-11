class VideoSplitSettings:
    def __init__(self,
                 small_chunk_per_second: float,
                 frame_per_small_chunk: int,
                 max_frame_seq_per_request: int,
                 max_small_chunk_per_message):
        if (frame_per_small_chunk % 2 != 1):
            raise Exception("frame_per_small_chunk must be an odd number in VideoSplitSettings!")
        
        self.small_chunk_per_second: float = small_chunk_per_second
        self.frame_per_small_chunk: int = frame_per_small_chunk
        self.max_frame_seq_per_request: int = max_frame_seq_per_request
        self.max_small_chunk_per_message: int = max_small_chunk_per_message

    def get_chunk_time_range(self, chunk_index) -> tuple[int, int]:
        '''
        Returns a tuple with start and end time of the small chunk calculated based on chunk_index.\n
        chunk_index: starts at 0
        '''
        start_t: float = self.seconds_per_small_chunk * chunk_index
        return (start_t, start_t+self.seconds_per_small_chunk)

    @property
    def frames_per_second(self):
        return self.small_chunk_per_second * self.frame_per_small_chunk
    @property
    def seconds_per_frame(self):
        return 1 / (self.small_chunk_per_second * self.frame_per_small_chunk)
    @property
    def seconds_per_small_chunk(self):
        return 1 / self.small_chunk_per_second
    
    @classmethod
    def defaultSettings(cls):
        return cls(1, 3, 30)