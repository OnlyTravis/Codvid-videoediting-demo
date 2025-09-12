class SmallChunk:
    def __init__(self,
                 id: int,
                 description: str,
                 frame_start: float,
                 frame_end: float):
        self.id: int = id
        self.description: str = description
        self.frame_start: float = frame_start
        self.frame_end: float = frame_end
    
    @property
    def duration(self):
        return self.frame_end+self.frame_start

class Chunk:
    def __init__(self,
                 id: int,
                 summary: str,
                 chunk_start: int,
                 chunk_end: int,
                 time_start: float,
                 time_end: float):
        self.id: int = id
        self.summary: int = summary
        self.chunk_start: int = chunk_start
        self.chunk_end: int = chunk_end
        self.time_start = time_start
        self.time_end = time_end
    
    @property
    def duration(self):
        return self.time_end+self.time_start