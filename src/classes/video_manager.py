import os
import shutil
import cv2
import uuid

class VideoManager:
    @classmethod
    def extract_all_frames(cls, video_path: str, interval_s: float) -> tuple[int, str]:
        """
        Extracts video frames with an interval to a folder in tmp.
        The extracted frames are stored as 0.jpg, 1.jpg ...
        Returns a tuple of number of frames extracted and the path to the folder
        """
        # 1. Setup frame extraction
        vidcap = cv2.VideoCapture(video_path)
        interval_ms = interval_s * 1000

        frame_folder = cls._create_frame_folder()
        print(f'Created Frame Folder: {frame_folder}')

        # 2. Extract frames
        count = 0
        while True:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, (count*interval_ms))    # added this line 
            success, image = vidcap.read()

            print(f'Reading Frame no. {count}: {"Success" if success else "Failed"}')
            if (not success): break

            cv2.imwrite(f"{frame_folder}/{count}.jpg", image)     # save frame as JPEG file
            count = count + 1
        
        # 3. Return folder name
        vidcap.release()
        print(f'End of extracing frames from {video_path}.')
        return (count, frame_folder)

    @classmethod
    def _create_frame_folder(cls) -> str:
        """
        Returns folder path of the created frame folder
        """
        folder_path = f'./tmp/{uuid.uuid4()}'
        os.mkdir(folder_path)
        return folder_path
    
    @classmethod
    def _remove_frame_folder(cls, folder_path: str) -> bool:
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
    
