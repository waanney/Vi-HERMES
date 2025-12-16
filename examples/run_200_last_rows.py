#!/usr/bin/env python3
"""
Script to run the 3-agent system on the last 200 rows of multihop_qa_dataset.csv
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from the same directory
from test_three_agents import evaluate_on_viquad


async def main():
    """Run evaluation on last 200 rows."""
    csv_path = Path(__file__).parent.parent / "data" / "multihop_qa_dataset.csv"
    output_path = Path(__file__).parent.parent / "result_200.csv"
    
    print("🚀 Running 3-Agent System on last 200 rows of multihop_qa_dataset.csv")
    print(f"📁 Input: {csv_path}")
    print(f"📁 Output: {output_path}")
    print()
    
    await evaluate_on_viquad(
        csv_path=str(csv_path),
        output_path=str(output_path),
        max_rows=200,
        top_k=5,
        batch_size=10,
        max_workers=5,
        reverse_order=True,  # Process from bottom to top
        use_context_block=True  # Use context_block from CSV instead of Milvus/Neo4j
    )
    
    print()
    print("✅ Done! Results saved to result_200.csv")
    print("📄 Summary saved to result_200_summary.txt")


if __name__ == "__main__":
    asyncio.run(main())

