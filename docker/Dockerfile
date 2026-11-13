FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Sanity check at build time: fail the build immediately (with a clear
# error) if the app package didn't get copied in correctly, instead of
# only finding out later at container start.
RUN python -c "import app.main" && echo "app.main import OK"

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
