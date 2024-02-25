FROM python:3.11 AS builder

ENV POETRY_VERSION=1.7.0

COPY pyproject.toml ./app/
COPY poetry.lock ./app/
COPY ./src ./app/src
COPY ./data ./app/data
WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV PYTHONPATH "${PYTHONPATH}:/app/src"

RUN pip3 install --upgrade pip
RUN apt-get update
RUN pip3 install "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.in-project true
RUN poetry install --with dev
RUN poetry run pip install torch-geometric==2.3.1


ENTRYPOINT ["poetry", "run", "streamlit", "run", "./src/webapp.py", "--server.port=8080"]
EXPOSE 8080
