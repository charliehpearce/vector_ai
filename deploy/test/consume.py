from kafka_pubsub.services import Kafka
from kafka_pubsub.client import ConsumerClient
from time import sleep
from json import loads


def callback(x):
    print(loads(x.decode('utf-8')))


service = Kafka(topic='picture_queue',
                bootstrap_servers='0.0.0.0:9092',
                group_id='new_group')

consumer = ConsumerClient(service=service, callback_fn=callback)
consumer.consume()
