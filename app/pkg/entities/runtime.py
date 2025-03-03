import asyncio
import datetime


class Timer:
    start_at: datetime.datetime
    processing: bool = False

    def __init__(self, task: asyncio.Task, start_at: datetime.datetime = None):
        self.task = task
        self.start_at = start_at or datetime.datetime.now()
