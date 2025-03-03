ifneq (,$(wildcard ./.env))
    include .env
    export
endif

# Список исключаемых папок
EXCLUDE = --exclude=dump --exclude=.git --exclude=.venv --exclude=.idea --exclude=resources/theme


ssh: ### Войти на сервер
	 ssh $(SERVER_USER)@$(SERVER_HOST)
.PHONY: ssh

sync:
	rsync -avz --delete $(EXCLUDE) ./ $(SERVER_USER)@$(SERVER_HOST):$(SERVER_PATH)
.PHONY: sync

build:
	ssh $(SERVER_USER)@$(SERVER_HOST) "docker build $(SERVER_PATH) -t chat-agent:$(tag)"
.PHONY: build

install:
	pipenv install
.PHONY: install

update:
	pipenv update
.PHONY: update

migrate_init:
	alembic revision --autogenerate -m "$(m)"
.PHONY: migrate_init

migrate:
	alembic upgrade head
.PHONY: migrate

migrate_downgrade:
	alembic downgrade -1
.PHONY: migrate_downgrade
