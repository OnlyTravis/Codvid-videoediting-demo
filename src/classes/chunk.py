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
    def __init__(self):
        pass