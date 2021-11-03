from kafka_pubsub.services import Kafka
from kafka_pubsub.client import ConsumerClient


def callback(x):
    print(x.decode('utf-8'))


service = Kafka(topic='results_queue',
                bootstrap_servers='0.0.0.0:9092',
                group_id='new_group')

consumer = ConsumerClient(service=service, callback_fn=callback)
consumer.consume()
