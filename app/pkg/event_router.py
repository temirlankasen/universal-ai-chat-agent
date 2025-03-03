import json
import ssl

import aiohttp
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type

from app.pkg.config import Config

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

TOOLS_GET_AUGMENTED_INFO = "get_augmented_info"


class EventRouter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = cfg.logger("event_router")

        self.handlers = {
            "get_": self._handle_get_info,
            "make_": self._handle_make_action,
        }

    async def route_event(self, event_name: str, external_id: str, user_id: str, user_type: str, **kwargs):
        # Если найден обработчик по префиксу, пытаемся выполнить его и ловим исключения,
        # чтобы гарантировать возврат ответа даже при неудачных ретраях.
        for prefix, handler in self.handlers.items():
            if event_name.startswith(prefix):
                try:
                    return await handler(external_id, event_name, user_id, user_type, **kwargs)
                except Exception as e:
                    self.logger.error(f"Ошибка обработки события '{event_name}': {e}")
                    return {"status": False, "message": f"Ошибка обработки события: {e}"}

        # Если обработчик не найден, возвращаем значение по умолчанию.
        return await self._default_handler(event_name)

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception))
    async def _handle_get_info(self, external_id: str, function_name: str, user_id: int, user_type: str, **kwargs):
        self.logger.debug(f"Обработка get info функции '{function_name}': external_id={external_id}, user_id={user_id}, user_type={user_type}")

        query = {
            "external_id": external_id,
            "user_id": user_id,
            "user_type": user_type,
            "event": function_name,
            "data": kwargs,
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(self.cfg.agent.dataset_url, ssl=ssl_context, headers={"Content-Type": "application/json"}, json=query) as response:
                status = response.status
                if status != 200:
                    raise Exception(f"Не удалось получить информацию, статус: {status}")

                self.logger.info(f"Получена информация для external_id {external_id}")
                try:
                    text = await response.text()
                    data = json.loads(text)
                except Exception:
                    data = {"status": status}
                return data

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception))
    async def _handle_make_action(self, external_id: str, function_name: str, user_id: int, user_type: str, **kwargs):
        self.logger.debug(f"Обработка make action функции '{function_name}': external_id={external_id}, user_id={user_id}, user_type={user_type}")

        payload = {
            "external_id": external_id,
            "user_id": user_id,
            "user_type": user_type,
            "event": function_name,
            "data": kwargs,
        }

        async with aiohttp.ClientSession() as session:
            url = f"{self.cfg.agent.dataset_url}/action"
            async with session.post(
                url,
                json=payload,
                ssl=ssl_context,
                headers={"Content-Type": "application/json"},
            ) as response:
                status = response.status
                if status != 200:
                    raise Exception(f"Не удалось выполнить действие, статус: {status}")

                self.logger.info(f"Действие выполнено для задачи {external_id}")
                try:
                    text = await response.text()
                    data = json.loads(text)
                except Exception:
                    data = {"status": status, "message": "Действие выполнено, но ответ не разобран"}
                return data

    async def _default_handler(self, event_name: str):
        self.logger.debug(f"Обработка события по умолчанию: {event_name}")
        return {"status": False, "result": "Неизвестный метод"}
