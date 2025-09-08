from src.extract_chunk import extract_chunks

def workflow(file_path: str, prompt: str): 
    # 1. Extract chunks with meaning from video
    chunks = extract_chunks(file_path)

    # 2. Generate video script with chunk descritions & prompt
    
    # 3. Prompt user to obtain final video script

    # 4. Assemble chunk list based on video script

    # 5. Export video
