from app.pkg.agents.base_agent import BaseAgent
from app.pkg.agents.openai_agent import OpenAIAgent
from app.pkg.agents.openai_assistant_agent import OpenAIAssistantAgent
from app.pkg.config import Config
from app.pkg.entities.models import AgentInstance
from app.pkg.llm.base_llm import BaseLLM
from app.pkg.vector_store.vector_store import VectorStore


def create_agent(agent_instance: AgentInstance, cfg: Config, vs: VectorStore, llm: BaseLLM) -> BaseAgent:
    if agent_instance.type == "openai":
        return OpenAIAgent(agent_instance, cfg, vs, llm)

    if agent_instance.type == "openai_assistant":
        return OpenAIAssistantAgent(agent_instance, cfg, vs, llm)

    raise ValueError(f"Неизвестный тип агента: {agent_instance.type}")
