from dataclasses import dataclass


@dataclass
class Google:
    subscription: str
    topic: str
    project: str


@dataclass
class Kafka:
    topic: str
    bootstrap_server: str
