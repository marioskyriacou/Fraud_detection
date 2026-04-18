FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Command to run the ML API on port 8000
CMD ["uvicorn", "ML_API:app", "--host", "0.0.0.0", "--port", "8000"]