# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Install system dependencies
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        libsqlite3-dev \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy the current directory contents into the container
COPY . /app

# Set the working directory
WORKDIR /app

# Expose port 5000 for the Flask app
EXPOSE 5000

# Run the application
CMD ["flask", "run"]