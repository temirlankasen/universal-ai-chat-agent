from app.pkg.config import Config
from app.pkg.llm.deepseek_llm import DeepSeekLLM
from app.pkg.llm.openai_llm import OpenAILLM


def create_llm(cfg: Config, llm_type: str) -> OpenAILLM | DeepSeekLLM:
    llm_choice = llm_type.lower()  # допустим, в конфигурации поле model задаёт "openai" или "deepseek"
    if llm_choice == "openai" or llm_choice == "openai_assistant":
        return OpenAILLM(cfg)
    elif llm_choice == "deepseek":
        return DeepSeekLLM(cfg)
    else:
        raise ValueError(f"Неизвестная модель LLM: {llm_choice}")
