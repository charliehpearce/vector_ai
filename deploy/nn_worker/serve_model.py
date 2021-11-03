import torch as t
from torch.serialization import load
from torchvision import transforms
from model import CovNet
from kafka_pubsub.services import Kafka
from kafka_pubsub.client import ConsumerClient, ProducerClient
import os
import io
from PIL import Image
from time import sleep

# This is to prevent a connection attempt before
# kafka is established
sleep(30)
print('Starting Model Server')

consume_service = Kafka(
    topic=os.environ['PIC_KAFKA_TOPIC'],
    bootstrap_servers=os.environ['KAFKA_BOOTSTRAP_SERVERS'],
    group_id=os.environ['GROUP_ID']
)

prod_service = Kafka(
    topic=os.environ['RESULT_KAFKA_TOPIC'],
    bootstrap_servers=os.environ['KAFKA_BOOTSTRAP_SERVERS'],
    group_id=None
)


def get_prediction(img_bytes, image_resize_shape):
    # transform image
    transformer = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(image_resize_shape),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
    # Read bytes image bytes and process into tensor
    print('opening bytes', flush=True)
    img = Image.open(io.BytesIO(img_bytes))
    print('turning into tensor', flush=True)
    img_tensor = transformer(img).unsqueeze(0)
    print('generting prediction', flush=True)
    outputs = model.forward(img_tensor)
    _, y_hat = outputs.max(1)
    return y_hat.item()


# Load model
image_resize_shape = (28, 28)
model_param_path = './model/fashion_mnst_cnn.bin'
model = CovNet(image_size=image_resize_shape)
model.load_state_dict(t.load(model_param_path))

# Shift to GPU if avalible
device = t.device('cuda' if t.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Open producer client
producer = ProducerClient(service=prod_service)


def process_image(img_bytes):
    pred = get_prediction(img_bytes=img_bytes,
                          image_resize_shape=image_resize_shape)
    print(pred)
    # Send to producer
    print('sending to results queue', flush=True)
    producer.send(bytes(str(pred), 'utf-8'))


print('Consuming picture', flush=True)
# Open consumer client and consume
consumer = ConsumerClient(consume_service, callback_fn=process_image)
consumer.consume()
