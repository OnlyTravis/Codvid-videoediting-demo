from src.extract_chunk import ExtractChunkSettings, extract_chunks

def workflow(file_path: str, user_prompt: str): 
    # 1. Extract chunks with meaning from video
    extract_settings = ExtractChunkSettings(
        small_chunk_per_second=0.67,
        frame_per_small_chunk=3,
        max_film_strip_per_message=30
    )
    chunks = extract_chunks(file_path, user_prompt, extract_settings)

    # 2. Generate video script with chunk descritions & prompt
    
    # 3. Prompt user to obtain final video script

    # 4. Assemble chunk list based on video script

    # 5. Export video
