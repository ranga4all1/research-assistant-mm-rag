# Use an official Python image as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy Pipfile and Pipfile.lock into the container
COPY Pipfile Pipfile.lock /app/

# Install pipenv to manage dependencies
RUN pip install pipenv

# Install dependencies from Pipfile
RUN pipenv install --system --deploy

# Copy the rest of the app's code to the container
COPY . /app

# Expose the port Gradio will run on
EXPOSE 7860

# Set environment variables for the Gradio app
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Command to run the Gradio app (actual script and app in the research-assistant folder)
CMD ["python", "research-assistant/gradio-app.py"]
