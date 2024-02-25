SHELL := /bin/bash

.DEFAULT_GOAL := help
.PHONY: help clean venv
IMAGE_NAME = "streamlit-app"
CONTAINER_NAME = "streamlit-container"

PACKAGE_NAME = "./src"
DATA_DIR = "./data"

# Miscilaneous commands
help: ## Print this help.
	@grep -E '^[0-9a-zA-Z%_-]+:.*## .*$$' $(firstword $(MAKEFILE_LIST)) awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-16s\033[0m %s\n", $$1, $$2}'

clean: ## Clean any artfacts and build components from system. Does not delete any downloaded data.
	@echo Removing .venv... & rm -rf .venv

venv: ## Create virtual environment using poetry.
	@poetry config virtualenvs.in-project true
	@poetry install --with dev
	@poetry config virtualenvs.in-project false

# data:
# 	@poetry run python -m src.data

build: ## Build docker image
	@docker build -t $(IMAGE_NAME) . --no-cache -f ./Dockerfile

# deploy: ## Deploy the streamlit app
# 	@docker container run -t -d --rm -p $(PORT):8080 \
# 	-v $(PWD)/$(PACKAGE_NAME):$(PACKAGE_NAME) \
# 	-v $(PWD)/$(DATA_DIR):$(DATA_DIR) \
# 	--name $(CONTAINER_NAME) $(IMAGE_NAME)
# 	@docker ps -s

deploy:
	@docker run -t -d --rm \
	-p 8080:8080 \
	--name $(CONTAINER_NAME) $(IMAGE_NAME)
	@docker ps -s

connect: ## Shell into running docker container.
	@docker container exec -it $(CONTAINER_NAME) /bin/bash

stop:  ## Stop docker container.
	@docker stop $(CONTAINER_NAME) || echo 'No container named $(CONTAINER_NAME) could be found'
