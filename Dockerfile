FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Run the application

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
