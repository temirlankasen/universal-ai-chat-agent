import asyncio
import ssl

import aiohttp

from app.pkg.config import Config

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

class WebSocket:
    def __init__(self, cfg: Config):
        self.cfg = cfg.ws
        self.logger = cfg.logger("ws")
        self.url = cfg.ws.url

    async def bot_typing(self, user_id: int, user_type: str):
        if not self.url:
            return

        async def post_request():
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": "apikey " + self.cfg.api_key}
                async with session.post(f"{self.cfg.url}/typing/{user_id}/{user_type}", ssl=ssl_context, headers=headers) as response:
                    self.logger.debug(f"Отправлено typing в канал {user_id}/{user_type}")
                    return response.status

        try:
            asyncio.create_task(post_request())
        except Exception as e:
            self.logger.error(f"Ошибка при отправке сообщения в канал {user_id}/{user_type}: {e}")

    async def start_typing(self, user_id: int, user_type: str):
        if not self.url:
            return

        while True:
            await asyncio.sleep(1)
            await self.bot_typing(user_id, user_type)
