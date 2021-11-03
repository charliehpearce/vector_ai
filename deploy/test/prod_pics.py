from kafka_pubsub.services import Kafka, Google
from kafka_pubsub.client import ProducerClient
import os
import time

kafka_service = Kafka(topic='picture_queue',
                      bootstrap_servers='0.0.0.0:9092',
                      group_id=None)

dir = './img'
lst = os.listdir(dir)
producer = ProducerClient(service=kafka_service)

for l in lst:
    image_path = os.path.join(dir, l)
    with open(image_path, 'rb') as f:
        img = f.read()
    producer.send(img)
    time.sleep(1)
