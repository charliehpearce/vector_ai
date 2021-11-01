from google.cloud import pubsub_v1
from kafka import KafkaProducer, KafkaConsumer
from services import Google, Kafka


class ProducerClient:
    def __init__(self, service) -> None:
        self.service = service
        self.service_type = type(service)

        if self.service_type == Google:
            self.publisher = pubsub_v1.PublisherClient()

        elif self.service_type == Kafka:
            self.publisher = KafkaProducer(
                bootstrap_servers=service.bootstrap_server)
        else:
            raise ValueError('Invalid Service')

    def send(self, message, **other_payload):
        if self.service_type == Google:
            future = self.publisher.publish(
                self.service.topic, message, **other_payload)
            id = future.result()
            print(f'Message published {id}')

        elif self.service_type == Kafka:
            future = self.publisher.send(
                self.service.topic, message, **other_payload)
            result = future.get(timeout=60)
            print(result)


class ConsumerClient:
    def __init__(self, service, callback_fn) -> None:
        self.callback_fn = callback_fn
        self.service = service

    def _g_callback(self, message):
        try:
            data = message.data
            self.callback_fn(data)
            message.ack()
        except:
            message.nack()

    def _g_consume(self):
        with pubsub_v1.SubscriberClient() as subscriber:
            future = subscriber.subscribe(
                self.service.subscription, callback=self._g_callback)
            try:
                future.result()
            except KeyboardInterrupt:
                future.cancel()

    def _k_consume(self):
        consumer = KafkaConsumer(
            self.service.topic,
            bootstrap_servers=self.service.bootstrap_servers,
            group_id=self.service.group_id,
        )

        for msg in consumer:
            self.callback_fn(msg.value)

    def consume(self):
        if type(self.service) == Google:
            print('Consuming Google')
            self._g_consume()
        elif type(self.service) == Kafka:
            self._k_consume()


if __name__ == "__main__":
    def callback(x):
        print(x.decode('utf-8'))

    service = Google(subscription='projects/canvas-advice-325410/subscriptions/test1',
                     topic='projects/canvas-advice-325410/topics/test1',
                     project='canvas-advice-325410')

    service = Kafka(
        topic='ahbshs',
        server='localhost'
    )

    consumer = ConsumerClient(service=service, callback_fn=callback)
    consumer.consume()
