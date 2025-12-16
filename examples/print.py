# ==========================================
# 1. NHẬP SỐ LIỆU CỦA BẠN TẠI ĐÂY
# ==========================================
avg_f1 = 0.7173
avg_recall = 0.7988
judge_status = False
judge_percent = 55.24
avg_tokens = 3957.19
avg_time = 14.85

# ==========================================
# 2. LỆNH PRINT (Không cần sửa phần này)
# ==========================================
print("=" *70)
print("📊 Final Summary")
print("=" * 70)
print(f"Average F1 Score: {avg_f1}")
print(f"Average Recall: {avg_recall}")
print(f"Average LLM Judge: {judge_status} ({judge_percent}%)")
print(f"Average Token Count: {avg_tokens}")
print(f"Average Time: {avg_time}s")