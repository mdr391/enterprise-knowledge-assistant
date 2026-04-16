#!/usr/bin/env python3
"""
Demo client — shows the full RAG pipeline end-to-end.
Ingests a document, then queries it with streaming output.

Prerequisites:
    - API running at localhost:8000
    - Knowledge base seeded: python scripts/seed_knowledge_base.py

Usage:
    python scripts/demo_query.py
    python scripts/demo_query.py --question "What is the on-call escalation path?"
    python scripts/demo_query.py --no-stream
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error


def query_stream(base_url: str, question: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  Question: {question}")
    print(f"{'─'*60}\n")

    payload = json.dumps({"question": question, "stream": True}).encode()
    req = urllib.request.Request(
        f"{base_url}/query/stream",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    sources = []
    print("  Answer: ", end="", flush=True)

    with urllib.request.urlopen(req, timeout=60) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                if "token" in data:
                    print(data["token"], end="", flush=True)
                elif "sources" in data:
                    sources = data["sources"]
                elif "error" in data:
                    print(f"\n\n  [ERROR] {data['error']}")
            except json.JSONDecodeError:
                pass

    print("\n")
    if sources:
        print(f"  Sources ({len(sources)}):")
        for s in sources:
            print(f"    • {s['title']}  [relevance: {s['relevance_score']:.2f}]")
    print()


def query_sync(base_url: str, question: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  Question: {question}")
    print(f"{'─'*60}\n")

    payload = json.dumps({"question": question, "stream": False}).encode()
    req = urllib.request.Request(
        f"{base_url}/query/",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())

    print(f"  Answer:\n  {data['answer']}\n")
    print(f"  Sources: {', '.join(s['title'] for s in data['sources'])}")
    print(f"  Latency: {data['latency_ms']}ms  |  Chunks: {data['retrieved_chunks']}\n")


DEMO_QUESTIONS = [
    "How many vacation days do I get after 3 years?",
    "What is the expense reimbursement limit for home office equipment?",
    "What is the on-call escalation path for a P0 incident?",
    "What are the rules for using LLMs in production?",
    "How do I enroll in benefits as a new employee?",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--question", default=None, help="Single question to ask")
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    questions = [args.question] if args.question else DEMO_QUESTIONS
    fn = query_sync if args.no_stream else query_stream

    print(f"\n{'═'*60}")
    print("  Enterprise Knowledge Assistant — Demo")
    print(f"{'═'*60}")

    for q in questions:
        try:
            fn(args.base_url, q)
            if len(questions) > 1:
                time.sleep(1)
        except urllib.error.URLError as e:
            print(f"\n[ERROR] Could not reach {args.base_url}: {e}")
            print("Make sure the API is running: uvicorn app.main:app --reload\n")
            sys.exit(1)
