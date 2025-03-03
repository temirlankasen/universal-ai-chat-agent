import datetime
import logging
import os

from sqlalchemy import update, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker

from app.pkg.entities.models import Base, Dialog, Message, AgentInstance


class Database:
    def __init__(
        self,
        logger: logging.Logger,
        url: str,
        name: str = "",
        read_only: bool = False,
        pool_size: int = 20,
        max_overflow: int = 5,
        pool_timeout: int = 5,
        pool_recycle: int = 30,
    ):
        """
        Инициализация подключения к БД с использованием асинхронного SQLAlchemy.
        """
        self.read_only = read_only
        self.logger = logger
        database_url = f"{url}/{name}"
        self.engine = create_async_engine(
            database_url,
            echo=os.environ.get("DEBUG_SQL", "0") == "1",
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
        )
        self.async_session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

    async def init_models(self):
        """
        Создание таблиц в БД (используется при первом запуске, до применения миграций).
        """
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def get_session(self) -> AsyncSession:
        """
        Получение нового асинхронного сеанса.
        """
        return self.async_session()

    async def save_agent_instance(
        self,
        name: str,
        agent_type: str,
        instruction: str,
        tools: list[str],
        response_format: dict,
        config: dict,
        meta: dict,
        cloud_id: str = None,
    ) -> AgentInstance:
        """
        Создание записи агента в БД.
        """
        async with self.async_session() as session:
            agent_instance = AgentInstance(
                name=name,
                type=agent_type,
                instruction=instruction,
                tools=tools,
                response_format=response_format,
                config=config,
                meta=meta,
                cloud_id=cloud_id,
            )

            session.add(agent_instance)
            await session.commit()
            await session.refresh(agent_instance)
            self.logger.debug(f"Агент создан с id: {agent_instance.id}")
            return agent_instance

    async def update_agent_instance(
        self,
        agent_id: int,
        name: str = None,
        agent_type: str = None,
        instruction: str = None,
        tools: list[str] = None,
        response_format: dict = None,
        config: dict = None,
        meta: dict = None,
        cloud_id: str = None,
    ):
        """
        Обновление данных агента.
        """
        async with self.async_session() as session:
            stmt = update(AgentInstance).where(AgentInstance.id == agent_id)
            update_data = {}
            if name:
                update_data["name"] = name
            if agent_type:
                update_data["type"] = agent_type
            if instruction:
                update_data["instruction"] = instruction
            if tools:
                update_data["tools"] = tools
            if response_format:
                update_data["response_format"] = response_format
            if config:
                update_data["config"] = config
            if meta:
                update_data["meta"] = meta
            if cloud_id:
                update_data["cloud_id"] = cloud_id
            if not update_data:
                return
            stmt = stmt.values(**update_data)
            await session.execute(stmt)
            await session.commit()
            self.logger.debug(f"Агент {agent_id} обновлён с данными: {update_data}")

    async def get_agent_instance(self, agent_id: int) -> AgentInstance:
        async with self.async_session() as session:
            stmt = select(AgentInstance).where(AgentInstance.id == agent_id)
            result = await session.execute(stmt)
            agent_instance = result.scalars().first()
            return agent_instance

    async def save_dialog(
        self,
        external_id: int,
        agent_id: int,
        user_id: int,
        user_type: str,
        meta: dict | None,
        started_at: datetime.datetime,
        thread_id: str = None,
        **kwargs,
    ) -> Dialog:
        """
        Создание записи чата в БД.
        """
        async with self.async_session() as session:
            dialog = Dialog(
                external_id=external_id,
                thread_id=thread_id,
                agent_id=agent_id,
                user_id=user_id,
                user_type=user_type,
                meta=meta,
                started_at=started_at,
            )

            session.add(dialog)
            await session.commit()
            await session.refresh(dialog)
            self.logger.debug(f"Чат создан с id: {dialog.id}")
            return dialog

    async def update_dialog(
        self,
        dialog_id: int,
        meta: dict = None,
        price: float = None,
        ended_at: datetime.datetime = None,
        last_message_at: datetime.datetime = None,
        user_id: int = None,
        thread_id: str = None,
    ):
        """
        Обновление данных чата.
        """
        async with self.async_session() as session:
            stmt = update(Dialog).where(Dialog.id == dialog_id)
            update_data = {}
            if user_id:
                update_data["user_id"] = user_id
            if meta:
                update_data["meta"] = meta
            if price:
                update_data["price"] = price
            if ended_at:
                update_data["ended_at"] = ended_at
            if last_message_at:
                update_data["last_message_at"] = last_message_at
            if thread_id:
                update_data["external_id"] = thread_id
            if not update_data:
                return
            stmt = stmt.values(**update_data)
            await session.execute(stmt)
            await session.commit()
            self.logger.debug(f"Диалог {dialog_id} обновлён с данными: {update_data}")

    async def create_message(self, message: Message) -> Message:
        """
        Создание сообщения (используется ORM-модель Message).
        """
        async with self.async_session() as session:
            session.add(message)
            await session.commit()
            await session.refresh(message)
            self.logger.debug(f"Создано сообщение с id: {message.id}")
            return message

    async def get_messages(self, dialog_id: int, before_id: int = None, after_id: int = None, limit: int = 10) -> list[Message]:
        """
        Получение сообщений чата с пагинацией.
        """
        async with self.async_session() as session:
            stmt = select(Message).where(Message.id == dialog_id)
            if before_id:
                stmt = stmt.where(Message.id < before_id)
            if after_id:
                stmt = stmt.where(Message.id > after_id)
            stmt = stmt.order_by(Message.id.desc()).limit(limit)
            result = await session.execute(stmt)
            messages = result.scalars().all()
            return messages

    async def get_active_dialogs(self) -> list[Dialog]:
        """
        Получение активных чатов (где ended_at IS NULL).
        """
        async with self.async_session() as session:
            stmt = select(Dialog).where(Dialog.ended_at.is_(None))
            result = await session.execute(stmt)
            dialogs = result.scalars().all()
            return dialogs

    async def insert(self, query: str, params: dict = None):
        """
        Выполняет произвольный SQL-запрос с параметрами для массовой вставки/обновления.
        Возвращает результат выполнения запроса.
        """
        async with self.async_session() as session:
            stmt = text(query)
            result = await session.execute(stmt, params)
            await session.commit()
            return result
