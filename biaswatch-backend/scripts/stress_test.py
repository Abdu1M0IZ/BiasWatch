from pathlib import Path
import argparse
import asyncio
import json
import statistics
import time

import httpx


OUTPUT_PATH = Path("artifacts/stress_test_results.json")

SAMPLE_TEXTS = [
    "you are so stupid",
    "hope you have a nice day",
    "this is a normal university update",
    "shut up you idiot",
    "i strongly disagree with your opinion",
]


async def send_request(client, url, text):
    start_time = time.perf_counter()

    try:
        response = await client.post(
            url,
            json={
                "text": text,
                "model_name": "best",
            },
            timeout=10.0,
        )

        latency = time.perf_counter() - start_time

        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "latency": latency,
        }

    except Exception as error:
        latency = time.perf_counter() - start_time

        return {
            "success": False,
            "status_code": None,
            "latency": latency,
            "error": str(error),
        }


async def run_stress_test(base_url, total_requests, concurrency):
    url = f"{base_url}/predict"

    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )

    async with httpx.AsyncClient(limits=limits) as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request(index):
            async with semaphore:
                text = SAMPLE_TEXTS[index % len(SAMPLE_TEXTS)]
                return await send_request(client, url, text)

        start_time = time.perf_counter()

        tasks = [
            bounded_request(index)
            for index in range(total_requests)
        ]

        results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time

    return results, total_time


def percentile(values, percent):
    sorted_values = sorted(values)
    index = int((percent / 100) * (len(sorted_values) - 1))
    return sorted_values[index]


def summarize_results(results, total_time, total_requests, concurrency):
    success_count = sum(1 for result in results if result["success"])
    latencies = [result["latency"] for result in results]

    summary = {
        "total_requests": total_requests,
        "concurrency": concurrency,
        "success_count": success_count,
        "failure_count": total_requests - success_count,
        "success_rate": round(success_count / total_requests, 4),
        "total_time_seconds": round(total_time, 4),
        "throughput_requests_per_second": round(total_requests / total_time, 4),
        "mean_latency_seconds": round(statistics.mean(latencies), 4),
        "median_latency_seconds": round(statistics.median(latencies), 4),
        "p95_latency_seconds": round(percentile(latencies, 95), 4),
        "status_codes": {},
    }

    for result in results:
        status_code = str(result["status_code"])
        summary["status_codes"][status_code] = (
            summary["status_codes"].get(status_code, 0) + 1
        )

    return summary


def save_results(summary):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print(f"saved stress test results to {OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=20)

    args = parser.parse_args()

    results, total_time = asyncio.run(
        run_stress_test(
            base_url=args.base_url,
            total_requests=args.requests,
            concurrency=args.concurrency,
        )
    )

    summary = summarize_results(
        results=results,
        total_time=total_time,
        total_requests=args.requests,
        concurrency=args.concurrency,
    )

    print(json.dumps(summary, indent=4))
    save_results(summary)


if __name__ == "__main__":
    main()