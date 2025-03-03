FROM python:3.13

RUN apt-get update && apt-get install -y \
    locales \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Aqtau

WORKDIR /project

RUN pip install --no-cache-dir pipenv
COPY Pipfile Pipfile.lock /project/
RUN pipenv install --deploy --system

COPY alembic /project/alembic
COPY alembic.ini /project/alembic.ini
COPY app /project/app

EXPOSE $PORT

CMD ["python", "-m", "app.cli.api"]