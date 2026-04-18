FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

# Command to run the Decision Engine API on port 8001
CMD ["uvicorn", "DE_API:app", "--host", "0.0.0.0", "--port", "8001"]