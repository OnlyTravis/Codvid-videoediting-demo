from math import floor
import cv2
import numpy as np
import os
import shutil
import uuid
import base64

from cv2.typing import MatLike

from src.tools.logger import Logger

class VideoHelper:
    @classmethod
    def extract_all_frames(cls, video_path: str, interval_s: float, max_size: int = -1) -> tuple[int, str]:
        """
        Extracts video frames with an interval to a folder in tmp.
        The extracted frames are stored as 0.jpg, 1.jpg ...
        Returns a tuple of number of frames extracted and the path to the folder
        """
        # 1. Setup frame extraction
        vidcap = cv2.VideoCapture(video_path)
        interval_ms = interval_s * 1000
        frame_folder = cls.create_frame_folder()
        Logger.log_print(f'Created Frame Folder: {frame_folder}')

        # 2. Check resolution
        w  = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        need_resize = (max_size != -1) and ((w > max_size) or (h > max_size))
        resize_w, resize_h = (max_size, floor(h * max_size / w)) if (w > h) else (floor(w * max_size / h), max_size)

        # 3. Extract frames
        count = 0
        Logger.log_print(f'Extracting Frames...')
        while True:
            # 3.1 Read frame
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count*interval_ms))
            success, image = vidcap.read()
            if (not success): break

            # 3.2 Check resize
            if (need_resize):
                image = cv2.resize(image, (resize_w, resize_h))
            
            # 3.3 Write to file
            cv2.imwrite(f"{frame_folder}/{count}.jpg", image)
            del image
            count = count + 1
        
        # 4. Return folder name
        vidcap.release()
        print(f'Finished extracing frames from {video_path}.')
        return (count, frame_folder)

    @classmethod    
    def merge_frames_to_frame_seq(cls, frame_list: list[MatLike]) -> MatLike:
        frame_seq = np.concatenate(frame_list, axis=1)
        return frame_seq
    
    @classmethod
    def video_path_to_base64(cls, video_path: str) -> str:
        video_file = open(video_path, 'rb')
        return base64.b64encode(video_file.read()).decode('utf-8')

    @classmethod    
    def matlike_to_base64(cls, data: MatLike, format: str = "jpeg", jpeg_quality: int = 85):
        if format.lower() == 'jpeg':
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            _, buffer = cv2.imencode(f".{format}", data, encode_params)
        else:
            _, buffer = cv2.imencode(f".{format}", data)
        
        base64_str = base64.b64encode(buffer).decode("utf-8")
        del buffer
        return base64_str

    @classmethod
    def matlike_to_base64_url(cls, data: MatLike, format: str = "jpeg", jpeg_quality: int = 85):
        b64_str = cls.matlike_to_base64(data, format, jpeg_quality)
        url = f"data:image/{format.lower()};base64,{b64_str}"

        del b64_str
        return url

    @classmethod
    def create_frame_folder(cls) -> str:
        """
        Returns folder path of the created frame folder
        """
        folder_path = f'./tmp/{uuid.uuid4()}'
        os.mkdir(folder_path)
        return folder_path
    
    @classmethod
    def remove_frame_folder(cls, folder_path: str) -> bool:
        """
        Removes the frame folder from tmp
        Return if the folder is successfully removed
        """
        if not os.path.isdir(folder_path):
            return False
        
        try:
            shutil.rmtree(folder_path)
            return True
        except:
            return False
    
