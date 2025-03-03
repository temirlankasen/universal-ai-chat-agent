from pydantic import BaseModel


class TypingRequest(BaseModel):
    user_id: int
    user_type: str


class OpenDialogRequest(BaseModel):
    user_id: int
    user_type: str
    meta: dict


class UpdateDialogRequest(BaseModel):
    user_id: int


class CreateMessageRequst(BaseModel):
    text: str | None = None
    media: list[dict] | None = None

    def not_text(self):
        return self.media is not None and self.media
