FROM python:3.9

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app/restful_routing_project

EXPOSE 8000

RUN pip list

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]