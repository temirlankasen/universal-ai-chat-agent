import datetime
from asyncio import Task
from typing import Any

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Numeric,
    ForeignKey,
    Text,
    func,
    Index,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship, reconstructor

from app.pkg.entities.api import CreateMessageRequst

Base = declarative_base()

MSG_SENDER_AGENT = "agent"
MSG_SENDER_USER = "user"
MSG_SENDER_SYSTEM = "system"


class ToDictMixin:
    def to_dict(self):
        return {column.key: getattr(self, column.key) for column in self.__table__.columns}


class Dialog(Base, ToDictMixin):
    __tablename__ = "dialog"

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(Integer, nullable=False, comment="ID чата во внешней системе")
    user_id = Column(Integer, nullable=False, comment="ID пользователя")
    user_type = Column(String, nullable=False, comment="Тип пользователя")
    agent_id = Column(Integer, ForeignKey("agent_instance.id", ondelete="CASCADE"), nullable=False, comment="ID агента")
    thread_id = Column(String, nullable=True, comment="ID потока, если агент предусматривает такую логику")
    started_at = Column(DateTime, default=func.now(), nullable=False, comment="Дата начала чата")
    ended_at = Column(DateTime, nullable=True, comment="Дата окончания чата")
    last_message_at = Column(DateTime, nullable=True, comment="Дата последнего сообщения")
    price = Column(Numeric, nullable=False, default=0, comment="Стоимость чата")
    meta = Column(JSONB, nullable=True, comment="Дополнительные параметры чата")

    messages = relationship("Message", back_populates="dialog", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_dialog_agent_id", "agent_id"),
        Index("ix_dialog_user", "user_id", "user_type"),
    )

    def __init__(self, **kw: Any):
        super().__init__(**kw)

        self.timer: Task | None = None
        self.is_processing: bool = False
        self.typing: Task | None = None
        self.runtime_messages: list = []
        self.queue: list = []
        self.has_not_text_messages: bool = False
        self.is_first_message: bool = True
        self.start_message_handle_at: datetime.datetime | None = None

    @reconstructor
    def init_on_load(self):
        # Метод, который вызывается после загрузки объекта из БД
        self.timer = None
        self.is_processing = False
        self.typing = None
        self.runtime_messages = []
        self.queue = []
        self.has_not_text_messages = False
        self.is_first_message = True
        self.start_message_handle_at = None

    def get_init_messages(self):
        return self.meta.get("init_messages", [])

    def add_queue(self, message: CreateMessageRequst):
        self.queue.append(message)
        if message.not_text():
            self.has_not_text_messages = True

    def add_runtime_message(self, message: str):
        self.runtime_messages.append(message)
        if len(self.runtime_messages) > 4:
            self.runtime_messages.pop(0)

    def get_personal_info(self):
        return self.meta.get("personal_info", "")

    @property
    def group(self):
        return self.meta.get("group", "")


class Message(Base):
    __tablename__ = "message"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dialog_id = Column(Integer, ForeignKey("dialog.id", ondelete="CASCADE"), nullable=False, index=True)
    sender = Column(String(50), nullable=False, comment="Тип отправителя сообщения. Пользователь, система или агент")
    text = Column(Text, nullable=True, comment="Текст сообщения")
    media = Column(
        JSONB, nullable=True, comment='Медиа-файлы. Формат: [{"mimetype": "image/jpeg", "type": "img", "url": "https://upload.com/some_image", "extension": "jpg", "name": "img_name", "size": 123}]'
    )
    created_at = Column(DateTime, server_default=func.now(), nullable=False, comment="Дата создания сообщения")
    meta = Column(JSONB, nullable=True, comment="Дополнительные параметры сообщения")

    dialog = relationship("Dialog", back_populates="messages")

    __table_args__ = (Index("ix_message_dialog_id", "dialog_id"),)


class AgentInstance(Base):
    __tablename__ = "agent_instance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, comment="Имя агента")
    type = Column(String, nullable=False, comment="Тип агента")
    instruction = Column(Text, nullable=True, comment="Основная инструкция для агента")
    consolidation_instruction = Column(Text, nullable=True, comment="Инструкция для консолидации в VectorStore")
    response_format = Column(JSONB, nullable=True, comment="Формат ответа")
    tools = Column(JSONB, nullable=True, comment="Доступные инструменты и методы")
    cloud_id = Column(String, nullable=True, comment="ID агента в облачной систем")
    config = Column(JSONB, nullable=True, comment="Конфигурация агента. Типичные настройки LLM")
    meta = Column(JSONB, nullable=True, comment="Дополнительные параметры агента")
    created_at = Column(DateTime, default=func.now(), nullable=False, comment="Дата создания агента")
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False, comment="Дата обновления агента")
