FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# Stage 2
FROM python:3.12-slim AS runner

WORKDIR /app

COPY --from=builder /app/venv venv
COPY app.py app.py

ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV FLASK_APP=app/app.py

EXPOSE 8080

CMD ["gunicorn", "--bind" , ":8080", "--workers", "2", "app:app"]