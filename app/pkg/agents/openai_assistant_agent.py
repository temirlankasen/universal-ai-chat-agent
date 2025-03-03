import asyncio
import json
import os
import tempfile

import aiohttp
import cv2
from moviepy import VideoFileClip
from openai import AsyncOpenAI

from app.pkg.agents.base_agent import BaseAgent
from app.pkg.config import Config
from app.pkg.entities.models import Dialog, AgentInstance
from app.pkg.event_router import TOOLS_GET_AUGMENTED_INFO
from app.pkg.llm import OpenAILLM
from app.pkg.llm.base_llm import BaseLLM
from app.pkg.vector_store.vector_store import VectorStore


class OpenAIAssistantAgent(BaseAgent):
    def __init__(self, agent_instance: AgentInstance, cfg: Config, vs: VectorStore, llm: BaseLLM|OpenAILLM):
        super().__init__(agent_instance, cfg, vs, llm)

        self.llm: OpenAILLM
        self.client = AsyncOpenAI(api_key=self.cfg.openai.api_key)

    async def build_asnwer(self, dialog: Dialog, question: str, media: list[dict] | None = None) -> tuple[str, list[str], dict]:
        messages = dialog.runtime_messages

        actions = []
        answer = ""
        is_success = False
        custom_data = {}
        run = None

        media_messages = await self._media_to_struct_message(media)
        if media_messages:
            for media_message in media_messages:
                if isinstance(media_message["content"], str):
                    messages.append(media_message["content"])

        instructions = dialog.meta.get("intruction", "")

        rag_instructions, augmentations = await self.vs.retrieval(self.agent_instance, messages, group=dialog.group)

        if augmentations:
            instructions = instructions + "\n\n" + rag_instructions
            custom_data["augmentations"] = augmentations

        try:
            await self.client.beta.threads.messages.create(thread_id=dialog.thread_id, role="user", content=question, timeout=3)
            self.logger.debug(f"Отправлен вопрос {dialog.id}: {question}")

            run = await self.client.beta.threads.runs.create_and_poll(
                thread_id=dialog.thread_id,
                assistant_id=self.agent_instance.cloud_id,
                additional_instructions=instructions,
                timeout=6,
                response_format=self.agent_instance.response_format,
                additional_messages=media_messages,
            )

            self.logger.debug(f"Создан run {dialog.id}: {run.id} - {run.status}: {question}")

            while not is_success:
                if run.status == "completed":
                    messages = await self.client.beta.threads.messages.list(thread_id=dialog.thread_id, limit=1, timeout=3)
                    result = json.loads(messages.data[0].content[0].text.value)

                    answer = result["answer"] if "answer" in result else ""
                    actions = [result["action"]] if "action" in result else []
                    result.pop("action", None)
                    result.pop("answer", None)
                    custom_data["response_data"] = result

                    self.logger.debug(f"Получен ответ run {dialog.id}: {run.id}: {answer}, {json.dumps(actions, ensure_ascii=False)}, {json.dumps(custom_data, ensure_ascii=False)}")
                    break
                elif run.status == "requires_action":
                    outputs = []

                    for call in run.required_action.submit_tool_outputs.tool_calls:
                        call_id = call.id
                        function = call.function
                        try:
                            data = json.loads(function.arguments)
                        except:
                            data = {}

                        if "events" not in custom_data:
                            custom_data["events"] = {}

                        self.logger.debug(f"run.status {dialog.id}: {run.status}. {question}. {function.name}")

                        if function.name == TOOLS_GET_AUGMENTED_INFO:
                            func_result, _ = await self.vs.retrieval(self.agent_instance, data.get("data", []), group=dialog.group, use_consolidation=False)
                        else:
                            func_result = await self.er.route_event(function.name, dialog.external_id, dialog.user_id, dialog.user_type, **data)

                        if not custom_data["events"].get(function.name):
                            custom_data["events"][function.name] = {}

                        custom_data["events"][function.name][call_id] = {
                            "data": data,
                            "func_result": func_result,
                        }

                        outputs.append({"tool_call_id": call_id, "output": json.dumps(func_result)})

                    try:
                        run = await self.client.beta.threads.runs.submit_tool_outputs_and_poll(
                            thread_id=dialog.thread_id,
                            run_id=run.id,
                            tool_outputs=outputs,
                            timeout=6,
                        )
                    except Exception as e:
                        self.logger.error(f"Ошибка отправки результата tools {dialog.id}: {e}")
                        run = await self.client.beta.threads.runs.poll(
                            thread_id=dialog.thread_id,
                            run_id=run.id,
                            timeout=6,
                        )
                elif run.status in ["expired", "cancelling", "cancelled", "failed"]:
                    try:
                        custom_data["code"] = run.last_error.code
                    except:
                        custom_data["code"] = run.status
                    custom_data["error"] = run.last_error.message
                    break
                elif run.status in ["incomplete", "queued", "in_progress"]:
                    run = await self.client.beta.threads.runs.poll(
                        thread_id=dialog.thread_id,
                        run_id=run.id,
                        timeout=6,
                    )
                else:
                    raise Exception("Неожиданное поведение openai")

        except Exception as e:
            self.logger.error(f"Ошибка создания ответа через gpt {dialog.id}: {e}")
            if run:
                try:
                    custom_data["code"] = run.last_error.code
                    custom_data["error"] = run.last_error.message
                except:
                    custom_data["code"] = 0
                    custom_data["error"] = str(e)
            else:
                custom_data["code"] = 0
                custom_data["error"] = str(e)

        return answer, actions, custom_data

    async def create_cloud(self) -> str:
        tools = self.agent_instance.tools
        main_tool = self.vs.main_tool()
        if main_tool:
            name = main_tool.get("function").get("name")
            tools_names = [tool.get("function").get("name") for tool in tools]

            if name not in tools_names:
                tools.append(main_tool)

        if self.agent_instance.cloud_id:
            try:
                response = await self.client.beta.assistants.update(
                    name=self.agent_instance.name,
                    assistant_id=self.agent_instance.cloud_id,
                    instructions=self.agent_instance.instruction,
                    model=self.cfg.openai.model,
                    tools=tools,
                    response_format=self.agent_instance.response_format,
                    temperature=self.agent_instance.config.get("temperature", 0.5),
                    top_p=self.agent_instance.config.get("top_p", 1.0),
                )
                self.logger.debug(f"Обновлен ассистент {self.agent_instance.id}: {response.id}")
                return response.id
            except Exception as e:
                self.logger.error(f"Ошибка обновления ассистента {self.agent_instance.id}: {e}")
        else:
            try:
                response = await self.client.beta.assistants.create(
                    name=self.agent_instance.name,
                    instructions=self.agent_instance.instruction,
                    model=self.cfg.openai.model,
                    tools=tools,
                    response_format=self.agent_instance.response_format,
                    temperature=self.agent_instance.config.get("temperature", 0.5),
                    top_p=self.agent_instance.config.get("top_p", 1.0),
                )
                self.logger.debug(f"Создан ассистент {self.agent_instance.id}: {response.id}")
                return response.id
            except Exception as e:
                self.logger.error(f"Ошибка создания ассистента {self.agent_instance.id}: {e}")

        return self.agent_instance.cloud_id if self.agent_instance.cloud_id else ""

    async def create_thread(self, dialog_id: int, init_messages: list = None) -> str | None:
        try:
            thread = await self.client.beta.threads.create(timeout=5, messages=init_messages)
            return thread.id
        except Exception as e:
            self.logger.error(f"Ошибка создания нового диалога {dialog_id}: {e}")
            return None

    async def delete_thread(self, dialog_id: int, thread_id: str | None) -> bool:
        if not thread_id:
            return True

        try:
            await self.client.beta.threads.delete(thread_id=thread_id, timeout=5)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка удаления диалога {dialog_id}: {e}")
            return False

    async def _media_to_struct_message(self, media: list[dict] | None):
        """
        Обрабатывает различные типы медиа (изображения, аудио, видео, файлы) для отправки в OpenAI Assistant.

        Args:
            media (dict, list[dict] или None): Медиа-словарь, список медиа-словарей или None.

        Returns:
            list[dict]: Структурированные сообщения, готовые к отправке в OpenAI Assistant.
        """
        if not media:
            return []

        messages = []

        for item in media:
            media_type = item.get('type')
            url = item.get('url')

            if not url:
                continue

            if media_type == 'img':
                # Скачиваем изображение и загружаем его в OpenAI
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as response:
                            if response.status != 200:
                                raise Exception(f"Не удалось скачать изображение с {url}")

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                                temp_img_filename = temp_file.name
                                temp_file.write(await response.read())

                    with open(temp_img_filename, "rb") as img_file:
                        file_id = await self.llm.save_file(img_file, purpose="vision")

                    if file_id:
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_file",
                                    "image_file": {
                                        "file_id": file_id
                                    }
                                }
                            ]
                        })
                    else:
                        # Если не удалось загрузить, пробуем использовать прямой URL
                        messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": url
                                    }
                                }
                            ]
                        })

                    # Удаляем временный файл
                    if os.path.exists(temp_img_filename):
                        os.remove(temp_img_filename)

                except Exception as e:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": url
                                }
                            }
                        ]
                    })

            elif media_type == 'audio':
                try:
                    transcribed_text = await self.llm.speech_to_text(url)
                    messages.append({
                        "role": "user",
                        "content": f"Транскрипция аудио: {transcribed_text}"
                    })
                except Exception as e:
                    messages.append({
                        "role": "user",
                        "content": f"Не удалось транскрибировать аудио с {url}. Ошибка: {str(e)}"
                    })

            elif media_type == 'video':
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url) as response:
                            if response.status != 200:
                                raise Exception(f"Не удалось скачать видеофайл с {url}")

                            # Создаем временный файл для хранения видео
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                                temp_video_filename = temp_file.name
                                # Записываем содержимое видео во временный файл
                                temp_file.write(await response.read())

                    # Извлекаем аудио из видео
                    video_clip = await asyncio.to_thread(VideoFileClip, temp_video_filename)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                        temp_audio_filename = temp_audio_file.name

                    await asyncio.to_thread(video_clip.audio.write_audiofile, temp_audio_filename)

                    transcribed_text = await self.llm.speech_to_text(temp_audio_filename, is_local_file=True)

                    messages.append({
                        "role": "user",
                        "content": f"Транскрипция аудио из видео: {transcribed_text}"
                    })

                    # Извлекаем кадры на 0%, 25%, 50%, 75% и 100% видео
                    cap = await asyncio.to_thread(cv2.VideoCapture, temp_video_filename)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    frame_positions = [0, 0.25, 0.5, 0.75, 1.0]
                    frame_descriptions = ["начала", "четверти", "середины", "трех четвертей", "конца"]

                    for i, pos in enumerate(frame_positions):
                        frame_idx = max(0, min(int(total_frames * pos), total_frames - 1))
                        await asyncio.to_thread(cap.set, cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = await asyncio.to_thread(cap.read)

                        if ret:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as frame_file:
                                frame_filename = frame_file.name

                            await asyncio.to_thread(cv2.imwrite, frame_filename, frame)

                            # Загружаем кадр в OpenAI
                            with open(frame_filename, "rb") as img_file:
                                file_id = await self.llm.save_file(img_file, purpose="vision")

                            if file_id:
                                position_name = frame_descriptions[i]
                                messages.append({
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"Кадр из {position_name} видео:"
                                        },
                                        {
                                            "type": "image_file",
                                            "image_file": {
                                                "file_id": file_id
                                            }
                                        }
                                    ]
                                })

                            os.remove(frame_filename)

                    await asyncio.to_thread(cap.release)

                    # Удаляем временные файлы
                    if os.path.exists(temp_audio_filename):
                        os.remove(temp_audio_filename)

                    if os.path.exists(temp_video_filename):
                        os.remove(temp_video_filename)

                except Exception as e:
                    messages.append({
                        "role": "user",
                        "content": f"Не удалось обработать видео с {url}. Ошибка: {str(e)}"
                    })

            elif media_type == 'file':
                # todo добавить чтение файлов
                messages.append({
                    "role": "user",
                    "content": f"Ссылка на файл: {url}"
                })

        return messages
