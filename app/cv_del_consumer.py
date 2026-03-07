import json
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


class ResumeDelConsumer:
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
            group_id="test02",
            value_deserializer=lambda m: json.loads(m),
            max_poll_interval_ms=1200000,
        )
        self.consumer.subscribe(["delete_cv"])

        self.del_api_url = f"http://0.0.0.0:{os.environ['PORT']}/api/resumes/delete"
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
                        payload = json.dumps({"cv_id": cv_id})

                        logger.info(f"Starting delete CV: {cv_id}")
                        response = requests.request(
                            "PUT",
                            self.del_api_url,
                            headers=self.headers,
                            data=payload,
                        )

                        logger.info(response)
                if self.stop_event.is_set():
                    return

            except Exception as e:
                logger.error(traceback.format_exc())
                # self.producer.send(topic=self.topic_send, value=info.results)

    def start(self):
        for _ in range(self.process):
            self.pool.apply_async(func=self.run)

        self.stop_event.wait()
        # self.pool.close()
        # self.pool.join()


def signal_handler(sig, frame):
    logger.info("Ctrl+C received ...")

    bi.stop_event.set()
    bi.pool.close()
    bi.pool.join()

    logger.info("All threads are done. Exiting.")
    logger.info("All threads are done. Exiting.")


if __name__ == "__main__":
    bi = ResumeDelConsumer()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    bi.start()
