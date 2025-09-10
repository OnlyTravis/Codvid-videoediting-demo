from pydantic import BaseModel, Field

class SmallChunkSchema(BaseModel):
    # Allow llm to link the output to the input film_strip (Hopefully)
    name: str = Field(description="Name of the film strip, formatted as 'film_strip_<index>'")
    descriptions: str = Field(description="Detailed description what's happening in the film strip.")

class SmallChunksOutputSchema(BaseModel):
    output: list[SmallChunkSchema] = Field(description="Output of results from each and every film strip")