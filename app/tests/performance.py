import requests
import time
import statistics
import glob
import os
import json

from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


if os.environ.get("APP_ENV") != "production":
    load_dotenv("./.env")


CONCURRENCY = 1
MAX_REQUEST = 5
LOG_FILE = "benchmark_result.json"

cv_data_path = r"C:\Users\p-CPLinhlee\AppData\Local\Linhlee\linhtinh\CV_data_test"

cv_data_list = glob.glob(cv_data_path + "/*")
cv_data_list = cv_data_list[:MAX_REQUEST]
total_request = len(cv_data_list)
print("Total requests:", total_request)

# extract_api_url = f"http://0.0.0.0:{os.environ['PORT']}/api/resumes/extract"
extract_api_url = "http://16.163.183.185:9001/api/resumes/extract"

def request_model(file_path):
    files = [
        ("cv_file", (os.path.basename(file_path), open(file_path, "rb"), "text/plain"))
    ]
    headers = {}
    payload = {}

    start = time.time()
    response = requests.request(
        "POST", extract_api_url, headers=headers, data=payload, files=files
    )
    latency = time.time() - start

    data = response.json()
    data = data["service_resp"]

    prompt_eval_count = data["prompt_eval_count"]
    prompt_eval_duration = data["prompt_eval_duration"]
    eval_count = data["eval_count"]
    eval_duration = data["eval_duration"]

    tokens_per_sec = 0
    tokens_per_sec = eval_count / (eval_duration / 1e9)
    total_token = prompt_eval_count + eval_count

    log = {
        "file": os.path.basename(file_path),
        "latency": latency,
        "prompt_tokens": prompt_eval_count,
        "output_tokens": eval_count,
        "total_tokens": total_token,
        "prompt_duration": prompt_eval_duration,
        "generation_duration": eval_duration,
        "tokens_per_sec": tokens_per_sec,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log) + "\n")

    return latency, tokens_per_sec, total_token


def run_test():
    latencies = []
    token_speeds = []
    total_tokens = []

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [
            executor.submit(request_model, file_path) for file_path in cv_data_list
        ]

        for future in as_completed(futures):
            latency, tps, total_token = future.result()

            latencies.append(latency)
            token_speeds.append(tps)
            total_tokens.append(total_token)

            print(f"Latency: {latency:.2f}s | Token/s: {tps:.2f}")

    total_time = time.time() - start_time
    latencies.sort()
    p95 = latencies[int(len(latencies) * 0.95)]

    summary = {
        "concurrency": CONCURRENCY,
        "total_test_time": round(total_time, 2),
        "throughput_req_per_sec": round(total_request / total_time, 2),
        "avg_latency": round(statistics.mean(latencies), 2),
        "p50_latency": round(statistics.median(latencies), 2),
        "p95_latency": round(p95, 2),
        "avg_total_tokens": round(sum(total_tokens) / total_request, 2),
        "avg_tokens_per_sec": round(statistics.mean(token_speeds), 2),
    }

    with open("benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    run_test()
