"""
Evaluate RAG System on ViQuAD Dataset

This script:
1. Reads ViQuAD CSV file
2. Queries RAG system for each question
3. Calculates F1 and EM (Exact Match) scores
4. Writes results to new CSV with columns: urax_answer, f1_score, em_score
"""

from __future__ import annotations

import asyncio
import ast
import csv
import re
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from uraxlaw.Agents.engine import LLMClient
from uraxlaw.config.settings import get_settings
from uraxlaw.lawgraph.neo4j_client import Neo4jClient
from uraxlaw.lawrag.milvus_client import MilvusClient

load_dotenv()

# Import query functions from query_rag.py
from query_rag import query_milvus, query_neo4j, build_enhanced_prompt, setup_milvus_client, setup_neo4j_client


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove extra spaces, punctuation)."""
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation (optional - can be adjusted)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    if not text:
        return []
    # Normalize and split
    normalized = normalize_text(text)
    return normalized.split()


def calculate_f1_score(predicted: str, ground_truth: str) -> float:
    """
    Calculate F1 score based on token overlap.
    Normalizes both predicted and ground_truth before calculation.
    
    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    # Normalize both inputs before calculation
    predicted = normalize_text(predicted) if predicted else ""
    ground_truth = normalize_text(ground_truth) if ground_truth else ""
    
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0
    
    pred_tokens = set(tokenize(predicted))
    truth_tokens = set(tokenize(ground_truth))
    
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    
    # Calculate intersection
    common_tokens = pred_tokens & truth_tokens
    
    if not common_tokens:
        return 0.0
    
    # Precision: common / predicted
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0
    
    # Recall: common / ground_truth
    recall = len(common_tokens) / len(truth_tokens) if truth_tokens else 0.0
    
    # F1 score
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_em_score(predicted: str, ground_truth: str) -> float:
    """
    Calculate Exact Match (EM) score.
    
    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        
    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    if not predicted and not ground_truth:
        return 1.0
    
    pred_normalized = normalize_text(predicted)
    truth_normalized = normalize_text(ground_truth)
    
    return 1.0 if pred_normalized == truth_normalized else 0.0


def extract_ground_truth_answers(answers_str: str) -> List[str]:
    """
    Extract ground truth answers from answers column.
    
    Format: {'text': ['answer1', 'answer2'], 'answer_start': [pos1, pos2]}
    """
    if not answers_str or answers_str == '':
        return []
    
    try:
        # Parse the dictionary string
        answers_dict = ast.literal_eval(answers_str)
        if isinstance(answers_dict, dict) and 'text' in answers_dict:
            texts = answers_dict['text']
            if isinstance(texts, list):
                return [str(text) for text in texts if text]
            elif isinstance(texts, str):
                return [texts]
    except (ValueError, SyntaxError):
        # Try to extract text directly if format is different
        pass
    
    return []


async def query_rag_for_answer(
    milvus_client: MilvusClient,
    neo4j_client: Neo4jClient,
    llm_client: LLMClient,
    question: str,
    context: str = "",
    top_k: int = 5
) -> str:
    """
    Query RAG system and return answer.
    
    Args:
        milvus_client: Milvus client
        neo4j_client: Neo4j client
        llm_client: LLM client
        question: Question to ask
        context: Optional context from CSV document
        top_k: Number of results to retrieve
        
    Returns:
        Generated answer string
    """
    try:
        # Query KB (Milvus)
        kb_results = query_milvus(milvus_client, question, top_k=top_k)
        
        # Query KG (Neo4j)
        kg_results = query_neo4j(neo4j_client, question, kb_results=kb_results, limit=top_k)
        
        # Generate answer
        if not kb_results and not kg_results and not context:
            return "Không tìm thấy thông tin liên quan."
        
        prompt = build_enhanced_prompt(question, kb_results, kg_results, context=context)
        answer = llm_client.complete(prompt)
        
        return answer.strip() if answer else "Không thể tạo câu trả lời."
    
    except Exception as e:
        print(f"⚠️  Error querying RAG: {e}")
        return f"Lỗi: {str(e)}"


async def query_gpt_direct(
    llm_client: LLMClient,
    question: str
) -> str:
    """
    Query GPT directly without RAG and without context (for comparison).
    
    Args:
        llm_client: LLM client
        question: Question to ask
        
    Returns:
        Generated answer string
    """
    try:
        prompt = f"""You are a Vietnamese legal expert. Answer the following question concisely and accurately.

Question: {question}

Respond in Vietnamese."""

        answer = llm_client.complete(prompt)
        return answer.strip() if answer else "Không thể tạo câu trả lời."
    
    except Exception as e:
        print(f"⚠️  Error querying GPT directly: {e}")
        return f"Lỗi: {str(e)}"


def format_reference_answers_for_prompt(ground_truth_answers: List[str]) -> str:
    """Format ground truth answers for inclusion in the QA judge prompt."""
    if not ground_truth_answers:
        return "No reference answers are available."
    formatted = []
    for idx, answer in enumerate(ground_truth_answers, 1):
        formatted.append(f"{idx}. {answer}")
    return "\n".join(formatted)


async def judge_answer_quality(
    llm_client: LLMClient,
    question: str,
    predicted_answer: str,
    ground_truth_answers: List[str],
    system_label: str
) -> int:
    """
    Ask an LLM judge to decide if the predicted answer satisfies the QA task.
    Returns 1 for PASS, 0 for FAIL.
    """
    if not predicted_answer or predicted_answer.startswith("Lỗi"):
        return 0

    reference_text = format_reference_answers_for_prompt(ground_truth_answers)
    prompt = f"""You are a QA judge evaluating the answer quality of system {system_label}.

Information:
- Question: {question}
- System answer: {predicted_answer}
- Reference answers (multiple valid phrasings may exist):
{reference_text}

Evaluation criteria:
1. The answer must align with the reference information (no hallucinations).
2. The content must directly address the question.
3. Different wording is acceptable if the meaning matches the reference answers.
4. If the reference answers do not cover the question, accept only responses that explicitly state "no information found" (or equivalent).

Respond with a single uppercase word: PASS if the answer satisfies the criteria, FAIL otherwise."""

    try:
        verdict = llm_client.complete(prompt)
        if not verdict:
            return 0
        verdict_upper = verdict.strip().upper()
        return 1 if verdict_upper.startswith("PASS") else 0
    except Exception as e:
        print(f"⚠️  Error judging QA quality for {system_label}: {e}")
        return 0


async def evaluate_on_viquad(
    csv_path: str | Path,
    output_path: str | Path,
    max_rows: Optional[int] = None,
    start_row: int = 0
):
    """
    Evaluate RAG system on ViQuAD dataset.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path to output CSV file
        max_rows: Maximum number of rows to process (None for all)
        start_row: Starting row index (for resuming)
    """
    print("=" * 70)
    print("Evaluating RAG System on ViQuAD Dataset")
    print("=" * 70)
    print()
    
    # Setup clients
    print("🔧 Setting up clients...")
    settings = get_settings()
    
    try:
        milvus_client = setup_milvus_client(settings)
        print("✅ Milvus client ready")
    except Exception as e:
        print(f"❌ Failed to setup Milvus: {e}")
        return
    
    try:
        neo4j_client = setup_neo4j_client(settings)
        print("✅ Neo4j client ready")
    except Exception as e:
        print(f"❌ Failed to setup Neo4j: {e}")
        return
    
    try:
        llm_client = LLMClient(model=settings.llm_model, api_key=settings.openai_api_key)
        print("✅ LLM client ready")
    except Exception as e:
        print(f"❌ Failed to setup LLM: {e}")
        return
    
    print()
    
    # Read CSV
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"❌ CSV file not found: {csv_path}")
        return
    
    print(f"📖 Reading CSV: {csv_path}")
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total_rows = len(rows)
    print(f"📊 Total rows: {total_rows}")
    
    # Limit rows if specified
    if max_rows:
        rows = rows[:max_rows]
        print(f"📝 Processing {len(rows)} rows (limited)")
    else:
        print(f"📝 Processing all {len(rows)} rows")
    
    # Skip to start_row if specified
    if start_row > 0:
        rows = rows[start_row:]
        print(f"⏩ Starting from row {start_row}")
    
    print()
    
    if not rows:
        print("⚠️  No rows to process.")
        return
    
    # Prepare output writer with fieldnames
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    base_fieldnames = list(rows[0].keys())
    eval_cols = [
        'urax_answer',
        'f1_score',
        'em_score',
        'gpt_answer',
        'gpt_f1_score',
        'gpt_em_score',
        'rag_qa_pass',
        'gpt_qa_pass',
    ]
    
    if 'metadata' in base_fieldnames:
        metadata_idx = base_fieldnames.index('metadata')
        for col in eval_cols:
            if col not in base_fieldnames:
                base_fieldnames.insert(metadata_idx + 1, col)
    else:
        for col in eval_cols:
            if col not in base_fieldnames:
                base_fieldnames.append(col)
    
    # Process each row
    results = []
    successful = 0
    failed = 0
    
    with open(output_path, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=base_fieldnames)
        writer.writeheader()
        
        for idx, row in enumerate(rows, start=start_row + 1):
            question = row.get('question', '').strip()
            answers_str = row.get('answers', '')
            
            if not question:
                print(f"[{idx}/{total_rows}] ⚠️  Skipping row {idx}: No question")
                results.append({
                    **row,
                    'urax_answer': '',
                    'f1_score': 0.0,
                    'em_score': 0.0,
                    'gpt_answer': '',
                    'gpt_f1_score': 0.0,
                    'gpt_em_score': 0.0,
                    'rag_qa_pass': 0,
                    'gpt_qa_pass': 0
                })
                writer.writerow(results[-1])
                csv_file.flush()
                continue
            
            print(f"[{idx}/{total_rows}] Processing: {question[:60]}...")
            
            try:
                # Normalize question before querying
                normalized_question = normalize_text(question)
                
                # Get context from row if available
                context = row.get('context', '').strip()
                
                # Query RAG system (with context from CSV)
                predicted_answer = await query_rag_for_answer(
                    milvus_client,
                    neo4j_client,
                    llm_client,
                    question,  # Use original question for query
                    context=context,  # Add context from CSV
                    top_k=5
                )
                
                # Query GPT directly (without context for fair comparison)
                gpt_answer = await query_gpt_direct(
                    llm_client,
                    question  # Use original question only, no context
                )
                
                # Extract ground truth answers
                ground_truth_answers = extract_ground_truth_answers(answers_str)
                
                # Normalize ground truth answers
                normalized_ground_truth_answers = [normalize_text(gt) for gt in ground_truth_answers]
                
                # Calculate scores for RAG answer (use best match if multiple ground truth answers)
                if ground_truth_answers:
                    # Normalize predicted answer before calculating scores
                    normalized_predicted = normalize_text(predicted_answer)
                    f1_scores = [calculate_f1_score(normalized_predicted, gt) for gt in normalized_ground_truth_answers]
                    em_scores = [calculate_em_score(normalized_predicted, gt) for gt in normalized_ground_truth_answers]
                    
                    # Use maximum scores (best match)
                    f1_score = max(f1_scores) if f1_scores else 0.0
                    em_score = max(em_scores) if em_scores else 0.0
                else:
                    f1_score = 0.0
                    em_score = 0.0
                
                # Calculate scores for GPT answer
                if ground_truth_answers:
                    # Normalize GPT answer before calculating scores
                    normalized_gpt = normalize_text(gpt_answer)
                    gpt_f1_scores = [calculate_f1_score(normalized_gpt, gt) for gt in normalized_ground_truth_answers]
                    gpt_em_scores = [calculate_em_score(normalized_gpt, gt) for gt in normalized_ground_truth_answers]
                    
                    # Use maximum scores (best match)
                    gpt_f1_score = max(gpt_f1_scores) if gpt_f1_scores else 0.0
                    gpt_em_score = max(gpt_em_scores) if gpt_em_scores else 0.0
                else:
                    gpt_f1_score = 0.0
                    gpt_em_score = 0.0
                
                rag_qa_pass = await judge_answer_quality(
                    llm_client,
                    question,
                    predicted_answer,
                    ground_truth_answers,
                    system_label="RAG"
                ) if ground_truth_answers else 0

                gpt_qa_pass = await judge_answer_quality(
                    llm_client,
                    question,
                    gpt_answer,
                    ground_truth_answers,
                    system_label="GPT Direct"
                ) if ground_truth_answers else 0

                results.append({
                    **row,
                    'urax_answer': predicted_answer,
                    'f1_score': f1_score,
                    'em_score': em_score,
                    'gpt_answer': gpt_answer,
                    'gpt_f1_score': gpt_f1_score,
                    'gpt_em_score': gpt_em_score,
                    'rag_qa_pass': rag_qa_pass,
                    'gpt_qa_pass': gpt_qa_pass
                })
                writer.writerow(results[-1])
                csv_file.flush()
                
                successful += 1
                print(f"   ✅ RAG - F1: {f1_score:.4f}, EM: {em_score:.4f} | GPT - F1: {gpt_f1_score:.4f}, EM: {gpt_em_score:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results.append({
                    **row,
                    'urax_answer': f'ERROR: {str(e)}',
                    'f1_score': 0.0,
                    'em_score': 0.0,
                    'gpt_answer': f'ERROR: {str(e)}',
                    'gpt_f1_score': 0.0,
                    'gpt_em_score': 0.0,
                    'rag_qa_pass': 0,
                    'gpt_qa_pass': 0
                })
                writer.writerow(results[-1])
                csv_file.flush()
                failed += 1
            
            # Save progress every 10 rows
            if idx % 10 == 0:
                print(f"   💾 Progress saved...")
    
    if results:
        print()
        print(f"✅ Results written incrementally to: {output_path}")
    
    # Calculate and print statistics
    print()
    print("=" * 70)
    print("📊 Evaluation Summary")
    print("=" * 70)
    print(f"Total processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if results:
        # RAG scores
        f1_scores = [r.get('f1_score', 0.0) for r in results if isinstance(r.get('f1_score'), (int, float))]
        em_scores = [r.get('em_score', 0.0) for r in results if isinstance(r.get('em_score'), (int, float))]
        rag_pass_scores = [r.get('rag_qa_pass', 0) for r in results if isinstance(r.get('rag_qa_pass'), (int, float))]

        # GPT scores
        gpt_f1_scores = [r.get('gpt_f1_score', 0.0) for r in results if isinstance(r.get('gpt_f1_score'), (int, float))]
        gpt_em_scores = [r.get('gpt_em_score', 0.0) for r in results if isinstance(r.get('gpt_em_score'), (int, float))]
        gpt_pass_scores = [r.get('gpt_qa_pass', 0) for r in results if isinstance(r.get('gpt_qa_pass'), (int, float))]
        
        print("\n📊 RAG System Scores:")
        if f1_scores:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            print(f"  Average F1 Score: {avg_f1:.4f}")
        
        if em_scores:
            avg_em = sum(em_scores) / len(em_scores)
            print(f"  Average EM Score: {avg_em:.4f}")
        if rag_pass_scores:
            rag_pass_rate = sum(rag_pass_scores) / len(rag_pass_scores)
            print(f"  QA Pass Rate: {rag_pass_rate:.4f}")
        
        print("\n📊 GPT Direct Scores:")
        if gpt_f1_scores:
            avg_gpt_f1 = sum(gpt_f1_scores) / len(gpt_f1_scores)
            print(f"  Average F1 Score: {avg_gpt_f1:.4f}")
        
        if gpt_em_scores:
            avg_gpt_em = sum(gpt_em_scores) / len(gpt_em_scores)
            print(f"  Average EM Score: {avg_gpt_em:.4f}")
        if gpt_pass_scores:
            gpt_pass_rate = sum(gpt_pass_scores) / len(gpt_pass_scores)
            print(f"  QA Pass Rate: {gpt_pass_rate:.4f}")
    
    print("=" * 70)


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RAG system on ViQuAD dataset')
    parser.add_argument(
        '--input',
        type=str,
        default='data/viquad_full_levels-1 - Sheet1.csv',
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/f1_and_em_results.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Maximum number of rows to process (for testing)'
    )
    parser.add_argument(
        '--start-row',
        type=int,
        default=0,
        help='Starting row index (for resuming)'
    )
    
    args = parser.parse_args()
    
    await evaluate_on_viquad(
        csv_path=args.input,
        output_path=args.output,
        max_rows=args.max_rows,
        start_row=args.start_row
    )


if __name__ == "__main__":
    asyncio.run(main())

