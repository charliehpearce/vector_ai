FROM python:3.9.7
COPY . /usr/src/app
WORKDIR /usr/src/app
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install kafka_pubsub-0.1.0-py3-none-any.whl
CMD ["python3", "./serve_model.py"]