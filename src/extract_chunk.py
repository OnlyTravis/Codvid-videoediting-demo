from src.classes.chunk import Chunk
from src.classes.video_manager import VideoManager

def extract_chunks(video_path: str) -> list[Chunk]:
    # 1. Extract frames
    frame_count, frame_folder = VideoManager.extract_all_frames(video_path, 0.5)
    

    return []