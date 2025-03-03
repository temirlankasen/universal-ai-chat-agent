import argparse
import asyncio

from app.pkg.backend_client import BackendClient
from app.pkg.config import Config
from app.pkg.db import Database
from app.pkg.llm import create_llm
from app.pkg.vector_store.vector_store import VectorStore

cfg = Config()
cfg.server.debug = True


async def test():
    llm = create_llm(cfg, "openai")
    db = Database(
        logger=cfg.logger("db"),
        url=cfg.db.url,
        name=cfg.db.name,
        pool_size=cfg.db.pool_size,
        max_overflow=cfg.db.max_overflow,
        pool_timeout=cfg.db.pool_timeout,
        pool_recycle=cfg.db.pool_recycle,
    )
    agent_instance = await db.get_agent_instance(cfg.agent.id)

    vs = VectorStore(cfg, llm)
    bc = BackendClient(cfg)

    # dataset = await bc.get_dataset()
    #
    # await vs.build_dataset(dataset)
    result = await vs.retrieval(agent_instance, ["да как ты смеешь!!!"], use_consolidation=False)
    print(result)


async def main(args):
    if args.action == "test":
        await test()
    else:
        raise Exception("Неверный тип действия: " + args.action)


if __name__ == "__main__":
    _args_parser = argparse.ArgumentParser(description="cli commands")
    _args_parser.add_argument("--action", "-a", type=str, default="test", help="Action")
    _args_parser.add_argument("--param", "-p", type=str, default="", help="Param")
    _args = _args_parser.parse_args()

    asyncio.run(main(_args))
