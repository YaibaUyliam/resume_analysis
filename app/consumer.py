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


if os.environ.get("APP_ENV") != "production":
    load_dotenv("./.env")

# setup_logging()
# logger = logging.getLogger(__name__)
# logger.info(os.environ.get("APP_ENV"))


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
            max_poll_interval_ms=1200000,
        )
        self.consumer.subscribe(["extract_cv_request"])

        self.producer = KafkaProducer(
            bootstrap_servers=os.environ["KAFKA"].split(","),
            value_serializer=lambda v: json.dumps(v).encode(),
        )
        self.duplication_result_topic = "duplicated_cv"
        self.extract_result_topic = "extract_cv_result"

        self.check_duplication_api_url = (
            f"http://0.0.0.0:{os.environ['PORT']}/api/resumes/check-duplication"
        )
        self.extract_api_url = (
            f"http://0.0.0.0:{os.environ['PORT']}/api/resumes/extract-store"
        )
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
                            "POST",
                            self.check_duplication_api_url,
                            headers=self.headers,
                            data=payload,
                        )
                        check_duplication_resp = response.json()
                        # if (
                        #     check_duplication_res["check_result"]["is_duplicate"]
                        #     is True
                        # ):
                        if len(check_duplication_resp["check_result"]) > 0:
                            value_duplicated_cv_topic = {
                                "cv_id": cv_id,
                                "cv_url": cv_url,
                                "job_id": item.get("job_id"),
                                "duplicated_cv": check_duplication_resp["check_result"],
                            }
                            self.producer.send(
                                topic=self.duplication_result_topic,
                                value=value_duplicated_cv_topic,
                            )

                        else:
                            payload = json.dumps(
                                {
                                    "cv_data": check_duplication_resp["cv_data_converted"],   # fmt: skip
                                    "cv_embed": check_duplication_resp["emb_result"],
                                    "file_name": check_duplication_resp["filename"],
                                }
                            )
                            response = requests.request(
                                "POST",
                                self.extract_api_url,
                                headers=self.headers,
                                data=payload,
                            )
                            logger.info(response.json())
                            cv_extract_res = response.json()
                            cv_extract_res["cv_id"] = cv_id
                            cv_extract_res["job_id"] = item.get("job_id")

                            self.producer.send(
                                topic=self.extract_result_topic, value=cv_extract_res
                            )

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
    bi = ResumeConsumer()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    bi.start()
