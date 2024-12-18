FROM python:3.9
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]