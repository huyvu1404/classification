FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY . /app

# Disable development dependencies
ENV UV_NO_DEV=1

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app
RUN uv sync --locked

RUN mkdir -p auth data

CMD ["sh", "-c", "uv run streamlit run main.py \
  --server.address=${STREAMLIT_HOST} \
  --server.port=${STREAMLIT_PORT}"]