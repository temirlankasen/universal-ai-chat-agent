import json
from contextlib import asynccontextmanager
from typing import List

from alembic import command
from alembic.config import Config as AlembicConfig
from fastapi import FastAPI, BackgroundTasks
from fastapi import UploadFile, HTTPException, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from app.pkg.config import Config
from app.pkg.dialog_agent import DialogAgent
from app.pkg.entities.api import UpdateDialogRequest, CreateMessageRequst, OpenDialogRequest
from app.pkg.entities.models import Dialog, Message, MSG_SENDER_USER

cfg = Config()

dialog_agent = DialogAgent(cfg=cfg)
logger = cfg.logger("server")


def run_migrations():
    alembic_cfg = AlembicConfig("alembic.ini")
    command.upgrade(alembic_cfg, "head")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    print(
        json.dumps(
            cfg.model_dump(),
            indent=4,
        )
    )

    run_migrations()

    await dialog_agent.load()

    try:
        yield
    finally:
        await dialog_agent.unload()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def check_api_key(request, call_next):
    try:
        if cfg.server.api_key is not None:
            if "Authorization" in request.headers:
                api_key = request.headers["Authorization"].split(" ")[-1]
                if api_key != cfg.server.api_key:
                    return JSONResponse(status_code=401, content={"detail": "Неверный api_key"})
        return await call_next(request)
    except Exception as e:
        if cfg.server.debug:
            raise e

        return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервера"})


@app.post("/dialog/{external_id}/ask/async")
async def ask_async(external_id: int, message: CreateMessageRequst):
    try:
        message = Message(text=message.text, media=message.media, sender=MSG_SENDER_USER)
        message_id = await dialog_agent.add_new_message(external_id, message)

        return {"message_id": message_id}
    except Exception as e:
        if cfg.server.debug:
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dialog/{external_id}/ask/sync")
async def ask_sync(external_id: int, message: CreateMessageRequst):
    try:
        message = Message(text=message.text, media=message.media, sender=MSG_SENDER_USER)
        return await dialog_agent.add_new_message(external_id, message, False)
    except Exception as e:
        if cfg.server.debug:
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/dataset/search")
async def search(message: CreateMessageRequst):
    try:
        message = Message(text=message.text, media=message.media, sender=MSG_SENDER_USER)

        result = await dialog_agent.vector_store.retrieval(dialog_agent.agent_instance, [message.text], use_consolidation=False)
        return {"result": result}
    except Exception as e:
        if cfg.server.debug:
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dialog/{external_id}")
async def get_chat(external_id: int):
    dialog: Dialog | None = dialog_agent.dialogs_manager.get_dialog(external_id)
    if dialog is None:
        raise HTTPException(status_code=404, detail="Диалог не найден")

    return {
        "dialog_id": dialog.id,
        "external_id": dialog.external_id,
        "user": {
            "id": dialog.user_id,
            "type": dialog.user_type,
            "name": dialog.meta.get("name", "Пользователь"),
        },
        "thread_id": dialog.thread_id,
        "started_at": dialog.meta.get("started_at"),
        "ended_at": dialog.meta.get("ended_at"),
        "meta": dialog.meta,
        "price": dialog.price,
    }


@app.get("/dialogs")
async def get_chats():
    dialogs = []

    for dialog in dialog_agent.dialogs_manager.dialogs.values():
        dialogs.append(
            {
                "dialog_id": dialog.id,
                "external_id": dialog.external_id,
                "user": {
                    "id": dialog.user_id,
                    "type": dialog.user_type,
                    "name": dialog.meta.get("name", "Пользователь"),
                },
                "thread_id": dialog.thread_id,
                "started_at": dialog.meta.get("started_at"),
                "ended_at": dialog.meta.get("ended_at"),
                "meta": dialog.meta,
                "price": dialog.price,
            }
        )

    dialogs.sort(key=lambda x: x["dialog_id"], reverse=True)

    return dialogs


@app.get("/dialog/{external_id}/messages")
async def get_messages(external_id: int, before_id: int, after_id: int, limit: int = 10):
    dialog = dialog_agent.dialogs_manager.get_dialog(external_id)

    return await dialog_agent.db.get_messages(dialog_id=dialog.id, before_id=before_id, after_id=after_id, limit=limit)


@app.post("/dialog/{external_id}")
async def open_dialog(external_id: int, request: OpenDialogRequest):
    try:
        await dialog_agent.open_dialog(external_id, request)
    except Exception as e:
        if cfg.server.debug:
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/dialog/{external_id}")
async def close_dialog(external_id: int):
    try:
        await dialog_agent.close_dialog(external_id)
    except Exception as e:
        if cfg.server.debug:
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/dialog/{external_id}")
async def update_dialog(external_id: int, request: UpdateDialogRequest):
    try:
        await dialog_agent.update_dialog(external_id, request.user_id)
    except Exception as e:
        if cfg.server.debug:
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dialog/typing/{user_id}/{user_type}")
async def user_typing(user_id: int, user_type: str):
    try:
        dialog_id = dialog_agent.dialogs_manager.get_dialog_id_by_user(user_id, user_type)
        if dialog_id:
            dialog = dialog_agent.dialogs_manager.get_dialog(dialog_id)
            if dialog:
                await dialog_agent.message_processor.trigger_processing(dialog)
                return {"status": "ok"}
        raise HTTPException(status_code=404, detail="Диалог не найден")
    except Exception as e:
        if cfg.server.debug:
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agent/dataset/reload/state")
async def get_reload_state():
    return {"reload_state": dialog_agent.reload_state}


@app.post("/agent/dataset/reload/file/sync")
async def make_reload_sync(files: List[UploadFile] = File(..., max_length=300), group: str | None = None):
    # Проверка наличия файлов
    if not files:
        raise HTTPException(status_code=400, detail="Требуется хотя бы один файл")

    # Проверка формата всех файлов
    for file in files:
        allowed_extensions = ["txt", "json"]
        if file.filename.split(".")[-1] not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"Файл {file.filename} имеет неверное расширение. Допустимы только " + ", ".join(allowed_extensions))

    # Чтение и декодирование файлов
    if len(files) > 1:
        contents = []

        for file in files:
            content_bytes = await file.read()

            if file.filename.split(".")[-1] == "txt":
                contents.append(content_bytes.decode("utf-8"))
            if file.filename.split(".")[-1] == "json":
                contents.append(json.loads(content_bytes.decode("utf-8")))

    else:
        contents = (await files[0].read()).decode("utf-8")

    # Если загружен более чем один файл, возвращаем массив содержимого файлов
    await dialog_agent.reload_dataset(contents, group)

    return {"reload_state": dialog_agent.reload_state}


@app.post("/agent/dataset/reload/sync")
async def make_reload_sync():
    await dialog_agent.reload_dataset()

    return {"reload_state": dialog_agent.reload_state}


@app.post("/agent/dataset/reload/async")
async def make_reload_async(background_tasks: BackgroundTasks):
    background_tasks.add_task(dialog_agent.reload_dataset)
    return {"status": "ok"}


@app.put("/agent")
async def make_create_agent():
    cloud_id = await dialog_agent.agent.create_cloud()
    if cloud_id != dialog_agent.agent_instance.cloud_id:
        await dialog_agent.db.update_agent_instance(dialog_agent.agent_instance.id, cloud_id=dialog_agent.agent_instance.cloud_id)
    return {"status": "ok"}


@app.get("/status")
async def get_status():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["use_colors"] = True
    log_config["formatters"]["default"]["datefmt"] = "%Y-%m-%d %H:%M:%S,000"
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"]["use_colors"] = True
    log_config["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S,000"

    uvicorn.run("app.cli.api:app", host="0.0.0.0", port=cfg.server.port, log_level="debug" if cfg.server.debug else "error", log_config=log_config)
