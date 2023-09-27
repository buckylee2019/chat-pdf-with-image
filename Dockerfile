# Use an official Python base image from the Docker Hub
FROM python:3.10-slim

# Install browsers
RUN apt-get update && apt-get install build-essential -y
# Install utilities
RUN apt-get install -y curl wget git

# Declare working directory
WORKDIR /app

COPY requirements.txt requirements.txt

# Install any necessary packages specified in requirements.txt.
RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "/app/chat.py", "--server.port=8501", "--server.address=0.0.0.0"]