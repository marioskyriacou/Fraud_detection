FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the API runs on
EXPOSE 8001

# Command to run the core API (Adjust 'BE_API:app' if needed)
CMD ["uvicorn", "BE_API:app", "--host", "0.0.0.0", "--port", "8002"]