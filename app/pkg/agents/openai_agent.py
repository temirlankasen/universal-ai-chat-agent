from app.pkg.agents.base_agent import BaseAgent
from app.pkg.entities.models import Dialog


class OpenAIAgent(BaseAgent):

    async def delete_thread(self, dialog_id: int, thread_id: str | None) -> bool:
        pass

    async def create_thread(self, dialog_id: int, init_messages: list = None) -> str | None:
        pass

    async def create_cloud(self) -> str:
        pass

    async def build_asnwer(self, dialog: Dialog, question: str, media: list | None = None) -> tuple[str, list[str], dict]:
        pass
