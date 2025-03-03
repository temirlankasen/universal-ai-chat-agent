import asyncio
import json

from langchain_core.documents import Document
from langchain_postgres import PGVector
from pydantic import BaseModel, Field

from app.pkg.config import Config
from app.pkg.entities.models import AgentInstance
from app.pkg.llm.base_llm import BaseLLM


class ConsalidationResponseFormat(BaseModel):
    rag_questions: list[str] = Field(description="Список вопросов")


class VectorStore:
    def __init__(self, cfg: Config, llm: BaseLLM):
        self.cfg = cfg
        self.logger = cfg.logger("vector-store")
        self.llm = llm
        # Создаем синхронное подключение для создания расширения,
        # затем асинхронное для работы
        self.vector_store = PGVector(
            embeddings=llm.embeddings,
            collection_name=f"dataset_{cfg.agent.id}",
            connection=f"{cfg.db.sync_url}/{cfg.db.name}",
            async_mode=False,
            embedding_length=2000,
            use_jsonb=True,
        )
        self.vector_store.create_vector_extension()
        self.vector_store = PGVector(
            embeddings=llm.embeddings,
            collection_name=f"dataset_{cfg.agent.id}",
            connection=f"{cfg.db.url}/{cfg.db.name}",
            async_mode=True,
            embedding_length=2000,
            create_extension=False,
            use_jsonb=True,
        )

    async def _prepare_group(self, group: str | None):
        """
        Если group передан, удаляет документы с данным значением в metadata.
        Иначе – пересоздает всю коллекцию.
        """
        if group:
            await self.vector_store.adelete(filter={"group": group})
        else:
            await self.vector_store.adelete_collection()
            await self.vector_store.acreate_collection()

    async def build_qa_dataset(self, dataset_input: list[dict], group: str | None = None):
        """
        Принимает датасет (список словарей: id, theme, question, answer).
        Если group задан, добавляет его в metadata каждого документа и удаляет старые документы с этим group.
        """
        documents = []
        for item in dataset_input:
            metadata = {}
            if "theme" in item:
                metadata["group"] = item["theme"]
            if group:
                metadata["group"] = group

            document = Document(
                id=item["id"],
                page_content=item["question"] + ". " + item["answer"],
                metadata=metadata,
            )
            documents.append(document)

        await self._prepare_group(group)
        await self.vector_store.aadd_documents(documents=documents)

    async def build_ready_format_dataset(self, dataset_input: list[dict], group: str | None = None):
        """
        Принимает датасет (список словарей: id, page_content, metadata).
        """

        documents = []
        for key, item in enumerate(dataset_input):
            document_id = item.get("id")
            metadata = item.get("metadata", {})
            if not document_id:
                metadata["index"] = key
            else:
                metadata["index"] = document_id
            if group:
                metadata["group"] = group

            document = Document(
                id=document_id,
                page_content=item.get("page_content"),
                metadata=metadata,
            )
            documents.append(document)

        await self._prepare_group(group)
        await self.vector_store.aadd_documents(documents=documents)

    async def build_sql_dataset(self, dataset_input: str, group: str | None = None):
        """
        Продумать парсинг для SQL
        """
        pass

    async def build_docs_dataset(self, dataset_input: list[str], group: str | None = None):
        """
        Принимает список строк и создает документы для каждой строки.
        Подразумевается, что это уже разбитые данные.
        """
        documents = []
        for index, item in enumerate(dataset_input):
            metadata = {"doc_index": index}
            if group:
                metadata["group"] = group
            document = Document(id=f"doc_{index}", page_content=item, metadata=metadata)
            documents.append(document)

        await self._prepare_group(group)
        await self.vector_store.aadd_documents(documents=documents)

    async def build_txt_dataset(self, dataset_input: str, delimiter: str = "\n\n", group: str | None = None, chunk_size: int|None = None, overlap_size: int|None = None):
        """
        Принимает текстовый датасет в виде строки, разбивает его на чанки по разделителю,
        и сохраняет документы. Если group передан – добавляет его в metadata.
        """
        # todo добавить chunk_size, overlap_size
        chunks = dataset_input.split(delimiter)

        documents = []
        for index, chunk in enumerate(chunks):
            metadata = {"txt_index": index}
            if group:
                metadata["group"] = group
            document = Document(id=f"txt_{index}", page_content=chunk, metadata=metadata)
            documents.append(document)

        await self._prepare_group(group)
        await self.vector_store.aadd_documents(documents=documents)

    async def build_dataset(self, dataset_input: list[dict] | str | list[str], group: str | None = None):
        if len(dataset_input) == 0:
            raise ValueError("Датасет пустой")

        if isinstance(dataset_input, list):
            if isinstance(dataset_input[0], list):
                await self.build_ready_format_dataset(dataset_input, group)
            if isinstance(dataset_input[0], dict):
                if dataset_input[0].get("page_content"):
                    await self.build_ready_format_dataset(dataset_input, group)
                else:
                    await self.build_qa_dataset(dataset_input, group)
            if isinstance(dataset_input[0], str):
                await self.build_docs_dataset(dataset_input, group)

        if isinstance(dataset_input, str):
            if "CREATE TABLE" in dataset_input:
                await self.build_sql_dataset(dataset_input, group)
            else:
                await self.build_txt_dataset(dataset_input, group=group)

    async def retrieval(self, agent_instance: AgentInstance, messages: list, group: str | None = None, use_consolidation: bool = True) -> tuple[str, list[str]]:
        """
        При retrieval поиск ограничивается документами, у которых в metadata group соответствует переданному значению.
        """
        questions = messages
        max_tokens = agent_instance.config.get("vs_max_tokens", 512)
        k = int(agent_instance.config.get("vs_k", 3))

        if not agent_instance.config.get("vs_loaded"):
            return "", []

        if use_consolidation and agent_instance.consolidation_instruction:
            response_format = {"type": "json_schema", "json_schema": ConsalidationResponseFormat}

            llm_response = await self.llm.invoke(
                prompt=json.dumps({"questions": messages}),
                intruction=agent_instance.consolidation_instruction.strip(),
                response_format=response_format,
                max_tokens=max_tokens,
            )
            questions = llm_response.rag_questions if llm_response and llm_response.rag_questions else questions

        augmentations = {}

        tasks = [self.vector_store.asimilarity_search_with_score(query=q, k=k, filter={"group": group} if group else None) for q in questions]
        tasks_res = await asyncio.gather(*tasks)
        threshold = None
        for aug_list in tasks_res:
            for aug in aug_list:
                distance = aug[1]
                document = aug[0]
                if threshold and distance > threshold:
                    continue
                augmentations[document.id] = document.page_content

        instruction = ""
        if augmentations:
            answers_instruction = "**Используй данную информацию чтобы ответить на сообщение пользователя. Если данных не хватает — попроси уточнить вопрос:**\n\n---\n\n"
            answers_instruction += "\n\n".join(augmentations.values())
            instruction = answers_instruction

        return instruction, list(augmentations.values())

    @staticmethod
    def main_tool() -> dict | None:
        # Определение для retrieval тулзы. Если не передать, retrieval будет только в момент обработки вопроса ДО llm
        return None
