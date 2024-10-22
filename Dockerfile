# Dockerfile

# Use the official Python image
FROM python:3.8

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set the environment variable for Django
ENV PYTHONUNBUFFERED=1

# Run the Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
