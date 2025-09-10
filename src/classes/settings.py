class ExtractChunkSettings:
    def __init__(self,
                 small_chunk_per_second: float,
                 frame_per_small_chunk: int,
                 max_film_strip_per_message: int):
        if (frame_per_small_chunk % 2 != 1):
            raise Exception("frame_per_small_chunk must be an odd number in ExtractChunkSettings!")
        
        self.small_chunk_per_second: float = small_chunk_per_second
        self.frame_per_small_chunk: int = frame_per_small_chunk
        self.max_film_strip_per_message: int = max_film_strip_per_message

    @property
    def frames_per_second(self):
        return self.small_chunk_per_second * self.frame_per_small_chunk
    @property
    def seconds_per_frame(self):
        return 1 / (self.small_chunk_per_second * self.frame_per_small_chunk)
    
    @classmethod
    def defaultSettings(cls):
        return cls(1, 3, 30)