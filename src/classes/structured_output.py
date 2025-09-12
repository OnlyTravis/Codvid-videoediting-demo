from pydantic import BaseModel, Field

class SmallChunkSchema(BaseModel):
    # Allow llm to link the output to the input frame_sequence (Hopefully)
    name: str = Field(description="Name of the frame sequence, formatted as 'frame_sequence_<index>'")
    description: str = Field(description="Detailed description what's happening in the frame sequence.")

class SmallChunksOutputSchema(BaseModel):
    output: list[SmallChunkSchema] = Field(description="An array of output from each and every frame sequence")

class ChunkSchema(BaseModel):
    # Allow llm to link the output to the input frame_sequence (Hopefully)
    start: int = Field(description="Id of the first text in the group")
    end: int = Field(description="Id of the last text in the group")
    summary: str = Field(description="A summary of every text included in the group.")

class GroupChunksOutputSchema(BaseModel):
    output: list[ChunkSchema] = Field(description="An array of output of every groups")