import json
import logging
import traceback
import os
import time
import requests

import signal
from multiprocessing.pool import ThreadPool
from threading import Event, Lock

from kafka import KafkaConsumer, KafkaProducer
from dotenv import load_dotenv

from .core import setup_logging


if os.environ.get("APP_ENV") != "production":
    load_dotenv("./.env")

setup_logging()
logger = logging.getLogger(__name__)
logger.info(os.environ.get("APP_ENV"))


class ResumeConsumer:
    def __init__(self):
        time.sleep(20)
        logger.info("Starting ....")

        self.process = int(os.environ["PROCESS"])
        self.pool = ThreadPool(self.process)
        self.lock = Lock()
        self.stop_event = Event()

        self.consumer = KafkaConsumer(
            bootstrap_servers=os.environ["KAFKA"].split(","),
            auto_offset_reset=os.environ["OFFSET"],
            group_id=os.environ["GROUP_ID"],
            value_deserializer=lambda m: json.loads(m),
        )
        self.consumer.subscribe(["extract_cv_request"])

        self.producer = KafkaProducer(
            bootstrap_servers=os.environ["KAFKA"].split(","),
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        self.topic_send = "extract_cv_result"

        self.api_url = f"http://0.0.0.0:{os.environ['PORT']}/api/resumes/extract"
        logger.info(self.api_url)
        self.headers = {"Content-Type": "application/json"}

    def run(self):
        while True:
            try:
                with self.lock:
                    data = self.consumer.poll(timeout_ms=5000, max_records=1)
                    if len(data) == 0:
                        # logger.info("Data empty")
                        time.sleep(10)

                for _, items in data.items():
                    for item in items:
                        item: dict = item.value
                        logger.info(item)

                        cv_id = item.get("cv_id")
                        if os.environ.get("ENV", "production") == "production":
                            cv_url = item.get("local_url")
                        else:
                            cv_url = item.get("public_url")

                        logger.info(cv_url)
                        payload = json.dumps({"cv_url": cv_url, "cv_id": cv_id})
                        response = requests.request(
                            "POST", self.api_url, headers=self.headers, data=payload
                        )

                        logger.info(response.json())
                        results = response.json()
                        results["cv_id"] = cv_id
                        results["job_id"] = item.get("job_id")

                        self.producer.send(topic=self.topic_send, value=results)

                if self.stop_event.is_set():
                    return

            except Exception as e:
                logger.error(item["order_id"])
                logger.error(traceback.format_exc())
                # self.producer.send(topic=self.topic_send, value=info.results)

    def start(self):
        for _ in range(self.process):
            self.pool.apply_async(func=self.run)

        self.stop_event.wait()
        # self.pool.close()
        # self.pool.join()


def signal_handler(sig, frame):
    print("Ctrl+C received ...")

    bi.stop_event.set()
    bi.pool.close()
    bi.pool.join()

    print("All threads are done. Exiting.")
    logger.info("All threads are done. Exiting.")


if __name__ == "__main__":
    bi = ResumeConsumer()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    bi.start()
