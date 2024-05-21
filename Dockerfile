FROM python:3.11-slim
ADD requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt --no-cache-dir
ADD . /app
WORKDIR /app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]