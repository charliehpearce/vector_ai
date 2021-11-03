# Kafka / Google PubSub Client

A simple unified API for both Kafka and Google PubSub.

## Installation

To install, download the .whl from the /dist/ folder into any directory and install with pip:

```bash
python3 -m pip install kafka_pubsub-0.1.0-py3-none-any.whl
```

This will install the project dependancies (Kafka-python, google-cloud-pubsub).

## Usage

To use either Google PubSub or Kafka the service requirements must first be satisfied. Create a service object as below.

```python
from kafka_pubsub.services import Kafka, Google

kafka_service = Kafka(topic='test_topic',
                      bootstrap_server='localhost:9092',
                      group_id='group1')

google_service = Google(
    subscription='projects/myproject/subscriptions/sub1',
    topic='projects/myproject/topics/topic1',
    project='myproject',
)

```

Please note that to use Google PubSub, a service account and auth will need to be configured. For more information please follow [this link](https://cloud.google.com/pubsub/docs/reference/libraries).

Alternativly, services and clients can be found within ./kafka_pubsub/.

### Producer API

The producer API can be used as follows.

```python
from kafka_pubsub.client import ProducerClient

producer = ProducerClient(service=service)
producer.send(item)
```

### Consumer API

The consumer API has been established to accept a service and a callback function. When data is recieved, the callback is triggered and the data can be handeled. 

```python
from kafka_pubsub.client import ConsumerClient

def callback(data):
	do_something_with(data)
  
consumer = ConsumerClient(service=service, callback_fn=callback)
consumer.consume()
```

### Notes on serialisation

To allow for maximum flexibility, no method of serialization has been included. Outgoing data will need to be serialised and deseralised (within the callback function).