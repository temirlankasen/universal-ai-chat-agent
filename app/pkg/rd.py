import redis.asyncio as redis

from app.pkg.config import Config


# todo добавить для горизонтального масштабирования
class RedisClient:
    def __init__(self, cfg: Config):
        if not hasattr(cfg, "redis_url") or not cfg.redis_url:
            raise Exception("Redis URL не задан в конфигурации")
        self.redis = redis.Redis.from_url(cfg.redis_url)

    async def set_value(self, key: str, value, expire: int = None):
        await self.redis.set(key, value, ex=expire)

    async def get_value(self, key: str):
        return await self.redis.get(key)

    async def delete_key(self, key: str):
        await self.redis.delete(key)

    async def exists(self, key: str) -> bool:
        return await self.redis.exists(key)
