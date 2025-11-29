import time
from dataclasses import dataclass

@dataclass
class BenchmarkResult:
    total_time: float = 0.0
    generation_time: float = 0.0
    retrieval_time: float = 0.0

class MetricsTracker:
    def __init__(self, model_name="RAG Model"):
        self.name = model_name
        self.start_time = 0
        self.end_time = 0
        self.gen_start = 0
        self.gen_total = 0

    def start(self):
        self.start_time = time.perf_counter()

    def start_gen(self):
        self.gen_start = time.perf_counter()

    def end_gen(self):
        self.gen_total += (time.perf_counter() - self.gen_start)

    def stop(self):
        self.end_time = time.perf_counter()

    def print_report(self):
        total = self.end_time - self.start_time
        retrieval = max(0, total - self.gen_total)

        print(f"\n{'='*10} REPORT: {self.name} {'='*10}")
        print(f"⏱️  Total Latency:    {total:.4f}s")
        print(f"🤖 Generation Time:  {self.gen_total:.4f}s (Gemini Thinking)")
        print(f"🔎 Retrieval Time:   {retrieval:.4f}s (Searching)")
        print(f"{'='*40}\n")    