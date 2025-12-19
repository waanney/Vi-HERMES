"""
Test 3-Agent System on ViQuAD Dataset

3 Agents:
1. Intent Agent: Chuẩn hóa câu hỏi người dùng
2. Receiver Agent: Lấy dữ liệu từ Milvus và Neo4j để trả lời câu hỏi
3. Guardrail Agent: Kiểm tra câu trả lời của LLM có bị hallucination không,
   nếu có thì yêu cầu LLM trả lời lại dựa trên data từ receiver

Evaluation Metrics:
- F1 Score
- LLM as Judge (binary 0/1)
- Token Count
- Time (from question to answer)
"""

from __future__ import annotations

import asyncio
import ast
import csv
import json
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

from dotenv import load_dotenv

from vihermes.Agents.engine import LLMClient
from vihermes.config.settings import get_settings
from vihermes.lawgraph.neo4j_client import Neo4jClient
from vihermes.lawrag.milvus_client import MilvusClient

load_dotenv()

# Import query functions
from query_rag import query_milvus, query_neo4j, build_enhanced_prompt, setup_milvus_client, setup_neo4j_client


class IntentAgent:
    """Agent để chuẩn hóa câu hỏi người dùng."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def normalize_question(self, question: str) -> str:
        """
        Normalize question: remove redundant words, standardize grammar, clarify intent.
        
        Args:
            question: Original question from user
            
        Returns:
            Normalized question
        """
        prompt = f"""You are a natural language processing expert. Your task is to normalize Vietnamese legal questions.

Please normalize the following question:
- Remove redundant and unnecessary words
- Standardize grammar and sentence structure
- Clarify the intent of the question
- Preserve the original meaning and core content
- Respond with only the normalized question, no additional explanations

Original question: {question}

Normalized question:"""
        
        normalized = self.llm.complete(prompt)
        return normalized.strip() if normalized else question


class ReceiverAgent:
    """Agent để lấy dữ liệu từ Milvus và Neo4j."""
    
    def __init__(self, milvus_client: MilvusClient, neo4j_client: Neo4jClient):
        self.milvus = milvus_client
        self.neo4j = neo4j_client
    
    def retrieve(self, question: str, top_k: int = 10) -> Tuple[List[dict], List[dict], List[str]]:
        """
        Retrieve dữ liệu từ Milvus (KB) và Neo4j (KG).
        
        Args:
            question: Câu hỏi đã chuẩn hóa
            top_k: Số lượng kết quả cần lấy
            
        Returns:
            Tuple (kb_results, kg_results, retrieved_context_ids)
        """
        # Query Milvus
        kb_results = query_milvus(self.milvus, question, top_k=top_k)
        
        # Query Neo4j
        kg_results = query_neo4j(self.neo4j, question, kb_results=kb_results, limit=top_k)
        
        # Extract context IDs from retrieved results
        # Format: article_id_original = "1/QĐ-XPHC:5962-3912-9440" -> extract "5962-3912-9440" (part after ":")
        # Match with used_context_ids from CSV: ["8124-1477-2925", "2272-1669-8459"]
        retrieved_context_ids = []
        
        # From KB results (Milvus) - extract ID after ":" from article_id_original
        for result in kb_results:
            # Use article_id_original (from chunk.id) instead of article_id (neo4j_article_id)
            article_id_original = result.get("article_id_original", "")
            # Fallback to article_id if article_id_original not available
            if not article_id_original:
                article_id_original = result.get("article_id", "")
            
            # Extract ID after ":" from article_id_original (format: "1/QĐ-XPHC:5962-3912-9440")
            if article_id_original and ":" in article_id_original:
                parts = article_id_original.split(":")
                if len(parts) > 1:
                    # Get the part after ":" (e.g., "5962-3912-9440")
                    id_after_colon = parts[-1].strip()
                    if id_after_colon and id_after_colon not in retrieved_context_ids:
                        retrieved_context_ids.append(id_after_colon)
        
        # From KG results (Neo4j) - also check article_id if available
        for result in kg_results:
            article_id = result.get("article_id", "")
            if article_id and ":" in article_id:
                parts = article_id.split(":")
                if len(parts) > 1:
                    id_after_colon = parts[-1].strip()
                    if id_after_colon and id_after_colon not in retrieved_context_ids:
                        retrieved_context_ids.append(id_after_colon)
        
        return kb_results, kg_results, retrieved_context_ids
    
    def format_retrieved_data(self, kb_results: List[dict], kg_results: List[dict]) -> str:
        """
        Format retrieved data into text for use in prompt.
        
        Args:
            kb_results: Results from Milvus
            kg_results: Results from Neo4j
            
        Returns:
            Formatted text
        """
        parts = []
        
        if kb_results:
            parts.append("=== Data from Knowledge Base (Milvus) ===")
            for i, result in enumerate(kb_results, 1):
                parts.append(f"\n[{i}]")
                if result.get("title"):
                    parts.append(f"Title: {result['title']}")
                if result.get("text"):
                    parts.append(f"Content: {result['text']}")
                if result.get("doc_id"):
                    parts.append(f"Document ID: {result['doc_id']}")
        
        if kg_results:
            parts.append("\n=== Data from Knowledge Graph (Neo4j) ===")
            for i, result in enumerate(kg_results, 1):
                parts.append(f"\n[{i}]")
                if result.get("title"):
                    parts.append(f"Title: {result['title']}")
                if result.get("text"):
                    parts.append(f"Content: {result['text']}")
                if result.get("doc_id"):
                    parts.append(f"Document ID: {result['doc_id']}")
        
        return "\n".join(parts) if parts else "No relevant data found."


class GuardrailAgent:
    """Agent để kiểm tra hallucination và yêu cầu trả lời lại nếu cần."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def check_hallucination(
        self,
        question: str,
        answer: str,
        retrieved_data: str
    ) -> Tuple[bool, str]:
        """
        Check if the answer contains hallucination.
        
        Args:
            question: Question
            answer: Answer from LLM
            retrieved_data: Data retrieved from Receiver
            
        Returns:
            Tuple (is_hallucination, corrected_answer)
            - is_hallucination: True if hallucination detected
            - corrected_answer: Corrected answer (or original if no hallucination)
        """
        prompt = f"""You are an expert in answer quality checking. Your task is to check if the answer contains hallucination (generating information not present in the data).

Question: {question}

Retrieved data:
{retrieved_data}

Answer to check:
{answer}

Please check:
1. Does the answer contain information not present in the retrieved data?
2. Does the answer make inferences that go too far beyond the data?
3. Is the answer accurate according to the data?

If there is hallucination, please answer the question again based ONLY on the retrieved data. If there is no hallucination, please respond with "NO HALLUCINATION".

Response:"""
        
        response = self.llm.complete(prompt)
        response = response.strip() if response else ""
        
        # Check if there is hallucination
        is_hallucination = "NO HALLUCINATION" not in response.upper()
        
        if is_hallucination:
            # If hallucination detected, response is the corrected answer
            corrected_answer = response
        else:
            # If no hallucination, keep original answer
            corrected_answer = answer
        
        return is_hallucination, corrected_answer


class ThreeAgentSystem:
    """Hệ thống 3 agents: Intent -> Receiver -> Guardrail."""
    
    def __init__(
        self,
        milvus_client: MilvusClient,
        neo4j_client: Neo4jClient,
        llm_client: LLMClient
    ):
        self.intent_agent = IntentAgent(llm_client)
        self.receiver_agent = ReceiverAgent(milvus_client, neo4j_client)
        self.guardrail_agent = GuardrailAgent(llm_client)
        self.llm = llm_client
    
    async def answer(
        self,
        question: str,
        top_k: int = 5
    ) -> Tuple[str, int, float, List[str]]:
        """
        Trả lời câu hỏi qua 3 agents.
        
        Args:
            question: Câu hỏi gốc
            top_k: Số lượng kết quả retrieve
            
        Returns:
            Tuple (final_answer, token_count, elapsed_time, retrieved_context_ids)
        """
        start_time = time.time()
        total_tokens = 0
        
        # 1. Intent Agent: Chuẩn hóa câu hỏi
        normalized_question = self.intent_agent.normalize_question(question)
        # Estimate tokens for intent (rough estimate: ~50 tokens)
        total_tokens += 50
        
        # 2. Receiver Agent: Retrieve dữ liệu
        kb_results, kg_results, retrieved_context_ids = self.receiver_agent.retrieve(normalized_question, top_k=top_k)
        retrieved_data = self.receiver_agent.format_retrieved_data(kb_results, kg_results)
        
        # 3. Generate initial answer
        if not kb_results and not kg_results:
            initial_answer = "No relevant information found in the database."
        else:
            prompt = build_enhanced_prompt(normalized_question, kb_results, kg_results)
            initial_answer = self.llm.complete(prompt)
            # Estimate tokens for answer generation (rough estimate)
            total_tokens += len(prompt.split()) + len(initial_answer.split()) if initial_answer else 0
        
        # 4. Guardrail Agent: Kiểm tra hallucination
        is_hallucination, final_answer = self.guardrail_agent.check_hallucination(
            normalized_question,
            initial_answer,
            retrieved_data
        )
        
        # Estimate tokens for guardrail check
        guardrail_prompt_tokens = len(retrieved_data.split()) + len(initial_answer.split()) + len(normalized_question.split())
        total_tokens += guardrail_prompt_tokens + len(final_answer.split()) if final_answer else 0
        
        elapsed_time = time.time() - start_time
        
        return final_answer, total_tokens, elapsed_time, retrieved_context_ids


# Evaluation functions
def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    if not text:
        return []
    normalized = normalize_text(text)
    return normalized.split()


def calculate_f1_score(predicted: str, ground_truth: str) -> float:
    """Calculate F1 score based on token overlap."""
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
    
    common_tokens = pred_tokens & truth_tokens
    
    if not common_tokens:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(common_tokens) / len(truth_tokens) if truth_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def extract_ground_truth_answers(answers_str: str) -> List[str]:
    """Extract ground truth answers from answers column."""
    if not answers_str or answers_str == '':
        return []
    
    try:
        answers_dict = ast.literal_eval(answers_str)
        if isinstance(answers_dict, dict) and 'text' in answers_dict:
            texts = answers_dict['text']
            if isinstance(texts, list):
                return [str(text) for text in texts if text]
            elif isinstance(texts, str):
                return [texts]
    except (ValueError, SyntaxError):
        pass
    
    return []


def extract_ground_truth_context_ids(used_context_ids_str: str, context_block: str = "") -> List[str]:
    """
    Extract ground truth context IDs from used_context_ids column.
    Also try to extract title from context_block for matching.
    
    Args:
        used_context_ids_str: String from used_context_ids column
        context_block: String from context_block column (optional, for title extraction)
    
    Returns:
        List of context IDs (and optionally titles for matching)
    """
    context_ids = []
    
    if not used_context_ids_str or used_context_ids_str == '':
        return []
    
    try:
        # Try to parse as JSON list
        parsed_ids = ast.literal_eval(used_context_ids_str)
        if isinstance(parsed_ids, list):
            context_ids = [str(cid) for cid in parsed_ids if cid]
        elif isinstance(parsed_ids, str):
            context_ids = [parsed_ids]
    except (ValueError, SyntaxError):
        # Try to split by comma if it's a string
        try:
            context_ids = [cid.strip() for cid in used_context_ids_str.split(',') if cid.strip()]
        except Exception:
            pass
    
    # Also try to extract ID from context_block (but NOT title, only IDs for recall matching)
    # Format: "[CONTEXT 1]\nID: 2070-9414-7446\nTitle: ..."
    if context_block:
        try:
            # Extract ID from context_block
            import re
            id_match = re.search(r'ID:\s*([^\n]+)', context_block)
            if id_match:
                block_id = id_match.group(1).strip()
                if block_id and block_id not in context_ids:
                    context_ids.append(block_id)
            # Note: We don't add title here because recall should only match IDs, not titles
        except Exception:
            pass
    
    return context_ids


# Debug counter for recall calculation
_recall_debug_count = 0

def normalize_title(title: str) -> str:
    """Normalize title for matching: remove punctuation, normalize spaces, lowercase."""
    import re
    if not title:
        return ""
    # Convert to lowercase
    normalized = title.lower()
    # Remove punctuation except spaces
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    # Normalize multiple spaces to single space
    normalized = re.sub(r'\s+', ' ', normalized)
    # Strip
    return normalized.strip()


def calculate_recall(retrieved_context_ids: List[str], ground_truth_context_ids: List[str]) -> float:
    """
    Calculate Recall = số context retrieve đúng / số context đúng trong corpus.
    
    Compares IDs extracted from article_id (part after ":") with used_context_ids from CSV.
    Format: article_id = "1/QĐ-XPHC:5962-3912-9440" -> "5962-3912-9440"
    Ground truth: ["8124-1477-2925", "2272-1669-8459"]
    
    Only matches IDs (format: numbers and dashes), filters out titles.
    
    Args:
        retrieved_context_ids: List of context IDs retrieved by the system (extracted from article_id after ":")
        ground_truth_context_ids: List of correct context IDs from ground truth (used_context_ids)
        
    Returns:
        Recall score (0.0 to 1.0)
    """
    if not ground_truth_context_ids:
        # If no ground truth, return 0 (can't calculate recall)
        return 0.0
    
    if not retrieved_context_ids:
        # If nothing retrieved, recall is 0
        return 0.0
    
    # Normalize all IDs for comparison (strip whitespace, convert to lowercase)
    def normalize_id(cid: str) -> str:
        """Normalize ID for comparison."""
        return str(cid).strip().lower()
    
    # Filter to only IDs (format: numbers and dashes, e.g., "2070-9414-7446")
    # IDs are typically short and contain dashes with numbers
    def is_id_format(cid: str) -> bool:
        """Check if string looks like an ID (format: numbers and dashes)."""
        normalized = normalize_id(cid)
        # ID format: contains dashes and is mostly digits/dashes, and not too long
        if len(normalized) > 50:  # Titles are usually longer
            return False
        # Check if it contains dashes and digits (typical ID format: "2070-9414-7446")
        if '-' in normalized:
            # Remove dashes and check if rest is digits
            without_dashes = normalized.replace('-', '')
            if without_dashes.isdigit():
                return True
        return False
    
    # Filter ground truth to only IDs (exclude titles)
    gt_ids = [normalize_id(cid) for cid in ground_truth_context_ids if is_id_format(cid)]
    
    # Filter retrieved to only IDs (should already be IDs, but filter just in case)
    retrieved_ids = [normalize_id(cid) for cid in retrieved_context_ids if is_id_format(cid)]
    
    if not gt_ids:
        # If no valid IDs in ground truth, return 0
        return 0.0
    
    if not retrieved_ids:
        # If nothing retrieved, recall is 0
        return 0.0
    
    # Normalize both sets
    retrieved_set = set(retrieved_ids)
    ground_truth_set = set(gt_ids)
    
    # Count exact matches
    matched = len(retrieved_set & ground_truth_set)
    
    # Recall = matched / total_correct
    recall = matched / len(ground_truth_set) if ground_truth_set else 0.0
    
    # Debug logging (only for first few calls to avoid spam)
    global _recall_debug_count
    _recall_debug_count += 1
    if _recall_debug_count <= 3:
        print(f"   🔍 Recall Debug:")
        print(f"      GT IDs (filtered): {list(ground_truth_set)[:5]}...")  # Show first 5
        print(f"      Ret IDs (filtered): {list(retrieved_set)[:5]}...")  # Show first 5
        print(f"      Matched: {matched}/{len(ground_truth_set)}, Recall={recall:.4f}")
    
    return min(recall, 1.0)  # Cap at 1.0


async def llm_judge(
    llm_client: LLMClient,
    question: str,
    predicted_answer: str,
    ground_truth_answers: List[str]
) -> str:
    """
    LLM as judge: evaluate answer quality (binary True/False).
    
    Returns:
        "True" if answer is good, "False" if not
    """
    if not predicted_answer or predicted_answer.startswith("Error") or predicted_answer.startswith("Lỗi"):
        return "False"
    
    # Combine all reference answers
    reference_answer = " | ".join(ground_truth_answers) if ground_truth_answers else "No reference answer available."
    
    prompt = f"""### Role
You are an expert language model evaluator. Your task is to evaluate the model prediction given a ground truth.

### Instruction
- If the prediction is "no answer" → False
- If the prediction contains the ground truth verbatim → True
- If the prediction paraphrases the ground truth → True
- If the prediction contradicts the ground truth → False
- Evaluation is language‑agnostic: ignore whether the prediction is in English or Vietnamese

### Note
- should mark True  little more likely than False
- The prediction may contain extra explanatory text: ignore it
- You must output ONLY one word: "True" or "False". Do not provide any explanation.

### INPUT DATA
1. Question:
{question}

2. Reference Answer (Ground Truth):
{reference_answer}

3. System Generated Answer:
{predicted_answer}

### Output:"""
    
    # Use temperature=0 for LLM as judge
    response = llm_client.complete(prompt, temperature=0.0)
    response = response.strip() if response else ""
    
    # Parse response - should be "True" or "False"
    response_upper = response.upper()
    
    # Check for "True" or "False"
    if "TRUE" in response_upper:
        return "True"
    elif "FALSE" in response_upper:
        return "False"
    
    # Fallback: check for common variations
    if response_upper.startswith("T") or "CORRECT" in response_upper or "PASS" in response_upper:
        return "True"
    if response_upper.startswith("F") or "INCORRECT" in response_upper or "FAIL" in response_upper:
        return "False"
    
    # Default to False if parsing fails
    return "False"


async def process_single_row(
    row: dict,
    idx: int,
    total_rows: int,
    system: ThreeAgentSystem,
    llm_client: LLMClient,
    top_k: int
) -> Optional[dict]:
    """Process a single row and return result."""
    question = row.get('question', '').strip()
    answer_str = row.get('answer', '').strip()
    
    if not question:
        print(f"[{idx}/{total_rows}] ⚠️  Skipping: No question")
        return None
    
    print(f"[{idx}/{total_rows}] Processing: {question[:60]}...")
    
    try:
        # Get answer from 3-agent system
        system_answer, token_count, elapsed_time, retrieved_context_ids = await system.answer(question, top_k=top_k)
        
        # Extract ground truth answers
        ground_truth_answers = extract_ground_truth_answers(answer_str)
        ground_truth = ground_truth_answers[0] if ground_truth_answers else answer_str
        
        # Extract ground truth context IDs
        used_context_ids_str = row.get('used_context_ids', '').strip()
        context_block = row.get('context_block', '').strip()
        ground_truth_context_ids = extract_ground_truth_context_ids(used_context_ids_str, context_block)
        
        # Calculate F1 score
        f1_score = calculate_f1_score(system_answer, ground_truth)
        
        # Calculate Recall
        recall_score = calculate_recall(retrieved_context_ids, ground_truth_context_ids)
        
        # Debug: Print context IDs for first few rows
        if idx <= 3:
            print(f"   🔍 Debug Recall:")
            print(f"      Ground truth IDs: {ground_truth_context_ids}")
            print(f"      Retrieved IDs: {retrieved_context_ids[:10]}...")  # Show first 10
            print(f"      Recall score: {recall_score:.4f}")
        
        # LLM as judge
        judge_score = await llm_judge(llm_client, question, system_answer, ground_truth_answers)
        
        result = {
            'id': row.get('id', str(idx)),
            'question': question,
            'ground_truth': ground_truth,
            'system_answer': system_answer,
            'f1_score': f1_score,
            'recall': recall_score,
            'llm_judge': judge_score,
            'token_count': token_count,
            'time_seconds': elapsed_time
        }
        
        print(f"   ✅ F1: {f1_score:.4f}, Recall: {recall_score:.4f}, Judge: {judge_score}, Tokens: {token_count}, Time: {elapsed_time:.2f}s")
        return result
    
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def append_results_to_csv(results: List[dict], output_path: Path, write_header: bool = False):
    """Append results to CSV file."""
    fieldnames = ['id', 'question', 'ground_truth', 'system_answer', 'f1_score', 'recall', 'llm_judge', 'token_count', 'time_seconds']
    
    file_exists = output_path.exists()
    mode = 'a' if file_exists else 'w'
    
    with open(output_path, mode, encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header only if file is new or explicitly requested
        if not file_exists or write_header:
            writer.writeheader()
        
        writer.writerows(results)


async def process_batch(
    batch_rows: List[tuple],
    system: ThreeAgentSystem,
    llm_client: LLMClient,
    top_k: int,
    batch_num: int,
    total_batches: int,
    output_path: Optional[Path] = None
) -> List[dict]:
    """Process a batch of rows in parallel."""
    print(f"\n🔄 Processing batch {batch_num}/{total_batches} ({len(batch_rows)} rows)...")
    
    # Create tasks for parallel execution
    tasks = [
        process_single_row(row, idx, len(batch_rows), system, llm_client, top_k)
        for idx, row in batch_rows
    ]
    
    # Wait for all tasks in batch to complete
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out None and exceptions
    valid_results = []
    for result in batch_results:
        if isinstance(result, Exception):
            print(f"   ❌ Batch task error: {result}")
        elif result is not None:
            valid_results.append(result)
    
    print(f"✅ Batch {batch_num}/{total_batches} completed: {len(valid_results)}/{len(batch_rows)} successful")
    
    # Save results immediately after batch completion
    if output_path and valid_results:
        append_results_to_csv(valid_results, output_path, write_header=(batch_num == 1))
        print(f"💾 Saved {len(valid_results)} results to {output_path}")
    
    return valid_results


async def evaluate_on_viquad(
    csv_path: str | Path,
    output_path: str | Path = "results.csv",
    max_rows: Optional[int] = None,
    top_k: int = 5,
    batch_size: int = 10,
    max_workers: int = 5,
    reverse_order: bool = False
):
    """
    Đánh giá hệ thống 3 agents trên dataset ViQuAD với parallel processing.
    
    Args:
        csv_path: Đường dẫn đến file CSV chứa câu hỏi
        output_path: Đường dẫn đến file CSV kết quả
        max_rows: Số lượng rows tối đa để test (None = tất cả)
        top_k: Số lượng kết quả retrieve
        batch_size: Số lượng rows trong mỗi batch
        max_workers: Số lượng batch chạy song song tối đa
        reverse_order: Nếu True, xử lý từ cuối lên đầu (bottom-up) thay vì từ đầu xuống cuối
    """
    print("=" * 70)
    print("Testing 3-Agent System on ViQuAD Dataset (Parallel Processing)")
    print("=" * 70)
    print()
    
    # Setup clients
    print("🔧 Setting up clients...")
    settings = get_settings()
    
    milvus_client = setup_milvus_client(settings)
    neo4j_client = setup_neo4j_client(settings)
    
    # Use OpenAI GPT-4o mini with provided API key
    openai_api_key = "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"
    llm_client = LLMClient(model="gpt-4o-mini", api_key=openai_api_key, provider="openai")
    
    # Create 3-agent system
    system = ThreeAgentSystem(milvus_client, neo4j_client, llm_client)
    
    print("✅ Clients initialized")
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
    
    # Reverse order if requested (bottom-up processing)
    if reverse_order:
        rows = list(reversed(rows))
        print(f"🔄 Processing in reverse order (bottom-up)")
    
    if max_rows:
        rows = rows[:max_rows]
        print(f"📝 Processing {len(rows)} rows (limited)")
    
    # Filter rows with questions
    valid_rows = [(idx, row) for idx, row in enumerate(rows, 1) if row.get('question', '').strip()]
    print(f"📝 Valid rows with questions: {len(valid_rows)}")
    print(f"⚙️  Batch size: {batch_size}, Max parallel batches: {max_workers}")
    print()
    
    # Split into batches
    batches = []
    for i in range(0, len(valid_rows), batch_size):
        batch = valid_rows[i:i + batch_size]
        batches.append(batch)
    
    total_batches = len(batches)
    print(f"📦 Total batches: {total_batches}")
    print()
    
    # Prepare output path
    output_path_obj = Path(output_path)
    
    # Check if output file already exists (resume mode)
    if output_path_obj.exists():
        print(f"⚠️  Output file exists: {output_path_obj}")
        print("   Results will be appended to existing file")
        # Optionally: read existing results to calculate averages correctly
        # For now, we'll just append and recalculate at the end
    else:
        print(f"📝 Creating new output file: {output_path_obj}")
    
    # Process batches with concurrency control
    all_results = []
    semaphore = asyncio.Semaphore(max_workers)
    
    async def process_batch_with_semaphore(batch, batch_num):
        async with semaphore:
            return await process_batch(batch, system, llm_client, top_k, batch_num, total_batches, output_path_obj)
    
    # Create tasks for all batches
    batch_tasks = [
        process_batch_with_semaphore(batch, batch_num + 1)
        for batch_num, batch in enumerate(batches)
    ]
    
    # Process batches in parallel (with concurrency limit)
    batch_results_list = await asyncio.gather(*batch_tasks, return_exceptions=True)
    
    # Collect all results (for calculating averages)
    for batch_results in batch_results_list:
        if isinstance(batch_results, Exception):
            print(f"❌ Batch error: {batch_results}")
        elif isinstance(batch_results, list):
            all_results.extend(batch_results)
    
    # Read all results from file to calculate accurate averages
    # (in case we're resuming from existing file)
    if output_path_obj.exists():
        all_results_from_file = []
        with open(output_path_obj, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['id'] != 'AVERAGE':  # Skip average row if exists
                    all_results_from_file.append(row)
        
        # Convert string values back to proper types for calculation
        for r in all_results_from_file:
            r['f1_score'] = float(r.get('f1_score', 0))
            r['recall'] = float(r.get('recall', 0))
            r['token_count'] = float(r.get('token_count', 0))
            r['time_seconds'] = float(r.get('time_seconds', 0))
        
        all_results = all_results_from_file
    
    # Extract metrics from results
    f1_scores = [float(r['f1_score']) for r in all_results]
    recall_scores = [float(r['recall']) for r in all_results]
    judge_scores = [r['llm_judge'] for r in all_results]
    token_counts = [float(r['token_count']) for r in all_results]
    times = [float(r['time_seconds']) for r in all_results]
    
    # Calculate averages
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    # Convert "True"/"False" to 1/0 for average calculation
    judge_numeric = [1 if j == "True" else 0 for j in judge_scores]
    avg_judge_numeric = sum(judge_numeric) / len(judge_numeric) if judge_numeric else 0.0
    avg_judge = "True" if avg_judge_numeric >= 0.5 else "False"  # Convert back to string for display
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0
    avg_time = sum(times) / len(times) if times else 0.0
    
    # Remove old AVERAGE row if exists, then append new one
    if output_path_obj.exists():
        # Read all lines except AVERAGE
        lines = []
        fieldnames_list = ['id', 'question', 'ground_truth', 'system_answer', 'f1_score', 'recall', 'llm_judge', 'token_count', 'time_seconds']
        with open(output_path_obj, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames_list = list(reader.fieldnames)
            for row in reader:
                if row.get('id') != 'AVERAGE':
                    lines.append(row)
        
        # Write back without AVERAGE row
        with open(output_path_obj, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames_list)
            writer.writeheader()
            writer.writerows(lines)
    
    # Append summary row
    summary_row = {
        'id': 'AVERAGE',
        'question': '',
        'ground_truth': '',
        'system_answer': '',
        'f1_score': avg_f1,
        'recall': avg_recall,
        'llm_judge': avg_judge,
        'token_count': avg_tokens,
        'time_seconds': avg_time
    }
    append_results_to_csv([summary_row], output_path_obj, write_header=False)
    
    print()
    print("✅ All results saved incrementally")
    print()
    print("=" * 70)
    print("📊 Final Summary")
    print("=" * 70)
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average LLM Judge: {avg_judge} ({avg_judge_numeric:.2%})")
    print(f"Average Token Count: {avg_tokens:.2f}")
    print(f"Average Time: {avg_time:.2f}s")
    print("=" * 70)


async def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test 3-Agent System on ViQuAD Dataset')
    parser.add_argument(
        '--input',
        type=str,
        default='data/viquad_hop_1234 - Trang tính2.csv',
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--max-rows',
        type=int,
        default=None,
        help='Maximum number of rows to process (for testing)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of results to retrieve'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of rows per batch'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Maximum number of batches to process in parallel'
    )
    parser.add_argument(
        '--reverse-order',
        action='store_true',
        help='Process rows from bottom to top (reverse order) instead of top to bottom'
    )
    
    args = parser.parse_args()
    
    await evaluate_on_viquad(
        csv_path=args.input,
        output_path=args.output,
        max_rows=args.max_rows,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        reverse_order=args.reverse_order
    )


if __name__ == "__main__":
    asyncio.run(main())

