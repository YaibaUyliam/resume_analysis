import json
import logging
import traceback
import os
import time
import requests
import signal

from loguru import logger
from multiprocessing.pool import ThreadPool
from threading import Event, Lock

from kafka import KafkaConsumer, KafkaProducer
from dotenv import load_dotenv

from .core import setup_logging


if os.environ.get("APP_ENV") != "production":
    load_dotenv("./.env")

# setup_logging()
# logger = logging.getLogger(__name__)
logger.info(os.environ.get("APP_ENV"))


class JDConsumer:
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
            group_id=os.environ["JD_GROUP_ID"],
            value_deserializer=lambda m: json.loads(m),
            max_poll_interval_ms=1200000,
        )
        self.consumer.subscribe(["recommend_cv_request"])

        self.producer = KafkaProducer(
            bootstrap_servers=os.environ["KAFKA"].split(","),
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        self.topic_send = "recommend_cv_result"

        self.api_url = f"http://0.0.0.0:{os.environ['PORT']}/api/jd/upload"
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

                        jd_id = item.get("id")

                        payload = json.dumps({"jd_content": item, "jd_id": jd_id})
                        response = requests.request(
                            "POST", self.api_url, headers=self.headers, data=payload
                        )

                        # logger.info(response.json())
                        results = response.json()
                        results["jd_id"] = jd_id
                        results["system_job_id"] = item.get("systemJobId")

                        self.producer.send(topic=self.topic_send, value=results)

                if self.stop_event.is_set():
                    return

            except Exception as e:
                logger.error(traceback.format_exc())

    def start(self):
        for _ in range(self.process):
            self.pool.apply_async(func=self.run)

        self.stop_event.wait()


def signal_handler(sig, frame):
    logger.info("Ctrl+C received ...")

    bi.stop_event.set()
    bi.pool.close()
    bi.pool.join()

    logger.info("All threads are done. Exiting.")
    logger.info("All threads are done. Exiting.")


if __name__ == "__main__":
    bi = JDConsumer()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    bi.start()
