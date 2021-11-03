# Deployment

To launch the deployment, run:

```bash
docker-compose up
```

The docker compose config is as below. More workers can be added by either changing the number of covnet_serve replicas or with `docker compose scale covnet_serve=n` after launch.

```yaml
version: '2'
services:
  zookeeper:
    image: wurstmeister/zookeeper:3.4.6
    ports:
      - "2181:2181"
    restart: "unless-stopped"
  
  kafka:
    build: ./kafka-docker
    ports:
      - "9092:9092"
    expose:
      - "9093"
    restart: "unless-stopped"
    environment:
      KAFKA_ADVERTISED_LISTENERS: INSIDE://kafka:9093,OUTSIDE://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: INSIDE:PLAINTEXT,OUTSIDE:PLAINTEXT
      KAFKA_LISTENERS: INSIDE://0.0.0.0:9093,OUTSIDE://0.0.0.0:9092
      KAFKA_INTER_BROKER_LISTENER_NAME: INSIDE
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_CREATE_TOPICS: "picture_queue:4:1, results_queue:4:1"
    volumes: 
      - /var/run/docker.sock:/var/run/docker.sock

  covnet_serve:
    restart: "unless-stopped"
    build: ./nn_worker
    deploy:
      replicas: 1
    environment:
      PIC_KAFKA_TOPIC: picture_queue
      KAFKA_BOOTSTRAP_SERVERS: kafka:9093
      GROUP_ID: "covnet_grp1" 
      RESULT_KAFKA_TOPIC: results_queue
    depends_on: 
      - "kafka"
```



Note: It's advisable to create a new network on every launch. Connection issues to Zookeeper tend to occur otherwise..



Scripts in `test/` can be used to help with testing.

