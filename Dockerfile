# Use an official Python runtime as a parent image
FROM python:3.9

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python -m venv /opt/venv

# Ensure the virtual environment is activated by modifying the PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy the current directory contents into the container at /code
COPY . /code/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Explicitly install ortools (if not in requirements.txt already)
RUN pip install ortools

# Expose the port the app runs on
EXPOSE 8000

# Debugging: Print installed packages
RUN pip list

# Run manage.py to start the Django app
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
