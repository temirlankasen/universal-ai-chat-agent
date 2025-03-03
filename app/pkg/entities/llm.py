from pydantic import BaseModel, Field


class AssistantConfig(BaseModel):
    temperature: float = Field(0.9)
    top_p: float = Field(0.8)
    response_format: dict
    instructions: str
    tools: list[dict]
