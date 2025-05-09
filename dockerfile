# Use an official Python runtime as a parent image
FROM python:3.13-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt /app/

# Install dependencies
RUN pip3.13 install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . /app

# Expose the port
EXPOSE 8000

# Start the server
CMD uvicorn main:app --host 0.0.0.0 --port $PORT