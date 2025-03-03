import json
import ssl

import aiohttp
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type

from app.pkg.config import Config

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


class BackendClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = cfg.logger("backend_client")

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception))
    async def get_dataset(self):
        if not self.cfg.agent.dataset_url:
            return []

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.cfg.agent.dataset_url}/reload?agent_id={self.cfg.agent.id}", ssl=ssl_context) as response:
                text_data = await response.text()
                try:
                    data = json.loads(text_data)
                    dataset = data.get("content", {}).get("dataset")
                    if not dataset:
                        raise Exception("Пустой датасет")
                    return dataset
                except Exception as e:
                    self.logger.error(f"Ошибка при парсинге датасета: {e}")
                    raise

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception))
    async def notify_reload_state(self, state: str):
        if not self.cfg.agent.dataset_url:
            return 200

        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.cfg.agent.dataset_url}/reload", json={"reload_state": state, "agent_id": self.cfg.agent.id}, ssl=ssl_context) as response:
                status = response.status
                if status != 200:
                    raise Exception(f"Не удалось уведомить о состоянии, статус: {status}")
                self.logger.info(f"Отправлен reload_state: {state}")
                return status

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception))
    async def send_answer(self, external_id: int, text: str, actions: list | None, custom_data: dict | None):
        if not self.cfg.agent.callback_url:
            return 200

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.cfg.agent.callback_url,
                data=json.dumps(
                    {
                        "agent_id": self.cfg.agent.id,
                        "external_id": external_id,
                        "text": text,
                        "actions": actions,
                        "custom_data": custom_data,
                    }
                ),
                ssl=ssl_context,
                headers={"Content-Type": "application/json"},
            ) as response:
                status = response.status
                if status != 200:
                    raise Exception(f"Не удалось отправить ответ, статус: {status}")
                self.logger.info(f"Отправлен ответ на задачу {external_id}")
                return status
