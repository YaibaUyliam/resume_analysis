from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_ENDPOINT = "http://127.0.0.1:8000/api/resumes/ingest"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".doc", ".xlsx"}


@dataclass(frozen=True)
class InputFile:
    path: Path
    content: bytes
    content_type: str


@dataclass
class RequestResult:
    request_id: int
    file_name: str
    status_code: int | None
    elapsed_ms: float
    stored: bool | None
    is_duplicate: bool | None
    duplicate_status: str | None
    error_detail: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the /api/resumes/ingest endpoint."
    )
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"Ingest endpoint URL. Default: {DEFAULT_ENDPOINT}",
    )
    parser.add_argument(
        "--input",
        default="app/tests/cv_corpus",
        help="A file or directory containing resume files to upload.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=10,
        help="Total number of requests to send.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Number of concurrent in-flight requests.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warm-up requests sent before measuring.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--cv-id-prefix",
        default="bench-cv",
        help="Prefix used to generate unique cv_id values.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional prompt file to include as multipart prompt_file.",
    )
    return parser.parse_args()


def discover_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported resume files found under: {input_path}")
    return files


def load_input_files(paths: list[Path]) -> list[InputFile]:
    loaded_files: list[InputFile] = []
    for path in paths:
        content_type, _ = mimetypes.guess_type(path.name)
        loaded_files.append(
            InputFile(
                path=path,
                content=path.read_bytes(),
                content_type=content_type or "application/octet-stream",
            )
        )
    return loaded_files


def build_file_payload(input_file: InputFile) -> tuple[str, bytes, str]:
    return (input_file.path.name, input_file.content, input_file.content_type)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


async def send_request(
    client: httpx.AsyncClient,
    endpoint: str,
    input_file: InputFile,
    request_id: int,
    cv_id_prefix: str,
    prompt_file: InputFile | None = None,
) -> RequestResult:
    data = {"cv_id": f"{cv_id_prefix}-{request_id:05d}"}
    files: dict[str, tuple[str, bytes, str]] = {"cv_file": build_file_payload(input_file)}
    if prompt_file is not None:
        files["prompt_file"] = build_file_payload(prompt_file)

    start = time.perf_counter()
    try:
        response = await client.post(endpoint, data=data, files=files)
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            body = response.json()
        except json.JSONDecodeError:
            body = {}

        return RequestResult(
            request_id=request_id,
            file_name=input_file.path.name,
            status_code=response.status_code,
            elapsed_ms=elapsed_ms,
            stored=body.get("stored"),
            is_duplicate=body.get("is_duplicate"),
            duplicate_status=body.get("duplicate_status"),
            error_detail=body.get("detail") if response.status_code >= 400 else None,
        )
    except Exception as exc:  # pragma: no cover - network/runtime failures
        elapsed_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            request_id=request_id,
            file_name=input_file.path.name,
            status_code=None,
            elapsed_ms=elapsed_ms,
            stored=None,
            is_duplicate=None,
            duplicate_status=None,
            error_detail=str(exc),
        )


async def run_phase(
    *,
    client: httpx.AsyncClient,
    endpoint: str,
    input_files: list[InputFile],
    total_requests: int,
    concurrency: int,
    cv_id_prefix: str,
    prompt_file: InputFile | None,
) -> tuple[list[RequestResult], float]:
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def worker(request_id: int) -> None:
        input_file = input_files[request_id % len(input_files)]
        async with semaphore:
            result = await send_request(
                client=client,
                endpoint=endpoint,
                input_file=input_file,
                request_id=request_id,
                cv_id_prefix=cv_id_prefix,
                prompt_file=prompt_file,
            )
            results.append(result)

    start = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(total_requests)))
    wall_time_s = time.perf_counter() - start
    results.sort(key=lambda item: item.request_id)
    return results, wall_time_s


def summarize_results(results: list[RequestResult], wall_time_s: float) -> str:
    latencies = sorted(result.elapsed_ms for result in results)
    success_results = [result for result in results if result.status_code == 200]
    error_results = [result for result in results if result.status_code != 200]

    status_histogram: dict[str, int] = {}
    duplicate_histogram: dict[str, int] = {}
    for result in results:
        status_key = str(result.status_code) if result.status_code is not None else "transport_error"
        status_histogram[status_key] = status_histogram.get(status_key, 0) + 1
        if result.duplicate_status:
            duplicate_histogram[result.duplicate_status] = (
                duplicate_histogram.get(result.duplicate_status, 0) + 1
            )

    throughput = len(results) / wall_time_s if wall_time_s > 0 else 0.0
    lines = [
        "Benchmark Summary",
        f"  total_requests: {len(results)}",
        f"  success: {len(success_results)}",
        f"  errors: {len(error_results)}",
        f"  wall_time_s: {wall_time_s:.2f}",
        f"  throughput_rps: {throughput:.2f}",
        f"  latency_ms_min: {min(latencies):.2f}" if latencies else "  latency_ms_min: 0.00",
        f"  latency_ms_mean: {statistics.mean(latencies):.2f}" if latencies else "  latency_ms_mean: 0.00",
        f"  latency_ms_p50: {percentile(latencies, 0.50):.2f}",
        f"  latency_ms_p95: {percentile(latencies, 0.95):.2f}",
        f"  latency_ms_p99: {percentile(latencies, 0.99):.2f}",
        f"  latency_ms_max: {max(latencies):.2f}" if latencies else "  latency_ms_max: 0.00",
        f"  stored_true: {sum(1 for result in success_results if result.stored is True)}",
        f"  duplicate_true: {sum(1 for result in success_results if result.is_duplicate is True)}",
        f"  duplicate_false: {sum(1 for result in success_results if result.is_duplicate is False)}",
        f"  status_codes: {json.dumps(status_histogram, sort_keys=True)}",
        f"  duplicate_statuses: {json.dumps(duplicate_histogram, sort_keys=True)}",
    ]

    if error_results:
        lines.append("  sample_errors:")
        for result in error_results[:5]:
            lines.append(
                f"    - request={result.request_id} status={result.status_code} file={result.file_name} detail={result.error_detail}"
            )
    return "\n".join(lines)


async def async_main(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    input_files = load_input_files(discover_input_files(input_path))
    prompt_file = None
    if args.prompt_file:
        prompt_path = Path(args.prompt_file).expanduser().resolve()
        prompt_file = load_input_files([prompt_path])[0]

    timeout = httpx.Timeout(args.timeout)
    limits = httpx.Limits(
        max_connections=max(args.concurrency, 1),
        max_keepalive_connections=max(args.concurrency, 1),
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        if args.warmup > 0:
            warmup_results, warmup_wall_time_s = await run_phase(
                client=client,
                endpoint=args.endpoint,
                input_files=input_files,
                total_requests=args.warmup,
                concurrency=min(args.concurrency, args.warmup),
                cv_id_prefix=f"{args.cv_id_prefix}-warmup",
                prompt_file=prompt_file,
            )
            print("Warmup Summary")
            print(summarize_results(warmup_results, warmup_wall_time_s))
            print()

        benchmark_results, benchmark_wall_time_s = await run_phase(
            client=client,
            endpoint=args.endpoint,
            input_files=input_files,
            total_requests=args.requests,
            concurrency=args.concurrency,
            cv_id_prefix=args.cv_id_prefix,
            prompt_file=prompt_file,
        )

    print(summarize_results(benchmark_results, benchmark_wall_time_s))
    return 0 if all(result.status_code == 200 for result in benchmark_results) else 1


def main() -> int:
    args = parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
