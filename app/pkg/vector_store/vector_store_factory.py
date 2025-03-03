from app.pkg.config import Config
from app.pkg.llm.base_llm import BaseLLM
from app.pkg.vector_store.vector_store import VectorStore


def create_vector_store(cfg: Config, llm: BaseLLM, preset: str) -> VectorStore:
        # todo если будут еще пресеты
        return VectorStore(cfg, llm)
