# 文件: src/tests/analyze_ia_cache.py
import json
import os
import sys

# ✅ 修正：从 src/tests/ 向上两级到达项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))  # src/tests/
src_dir = os.path.dirname(script_dir)  # src/
project_root = os.path.dirname(src_dir)  # ReviewMirabel/

# 添加 src 到 Python 路径
sys.path.insert(0, src_dir)

# ✅ cache 文件夹与 src 平级
cache_path = os.path.join(project_root, "cache", "ia_test_fixed.json")


def analyze_unknown_ratio():
    """分析 RAG 响应中的 unknown 比例"""

    print(f"[Debug] Script dir: {script_dir}")
    print(f"[Debug] Src dir: {src_dir}")
    print(f"[Debug] Project root: {project_root}")
    print(f"[Debug] Cache path: {cache_path}")
    print()

    if not os.path.exists(cache_path):
        print(f"❌ Cache file not found: {cache_path}")
        print(f"   Current working directory: {os.getcwd()}")

        # 列出 cache 目录内容
        cache_dir = os.path.join(project_root, "cache")
        if os.path.exists(cache_dir):
            print(f"\n   Files in cache directory:")
            for f in os.listdir(cache_dir):
                print(f"     - {f}")
        else:
            print(f"   Cache directory does not exist: {cache_dir}")

        return

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    print("=" * 80)
    print("IA RAG Response Analysis - Unknown Ratio")
    print("=" * 80)

    rag_keys = [k for k in cache.keys() if k.startswith("ia_rag_")]

    if not rag_keys:
        print("❌ No RAG response entries found in cache")
        return

    print(f"\nFound {len(rag_keys)} RAG response entries\n")

    total_unknown = 0
    total_responses = 0

    for key in sorted(rag_keys):
        responses = cache[key]["parsed_responses"]
        unknown_count = responses.count("unknown")
        yes_count = responses.count("yes")
        no_count = responses.count("no")
        total = len(responses)

        total_unknown += unknown_count
        total_responses += total

        ratio = unknown_count / total if total > 0 else 0

        # 提取文档 ID
        doc_id = key.split("_")[2]

        # 使用颜色标记问题严重程度
        if ratio > 0.8:
            status = "🔴"  # 严重问题
        elif ratio > 0.5:
            status = "🟡"  # 中等问题
        else:
            status = "🟢"  # 正常

        print(f"{status} Doc {doc_id:6s}: {unknown_count:2d}/{total} = {ratio:6.2%} unknown "
              f"(yes: {yes_count:2d}, no: {no_count:2d})")

    # 总体统计
    overall_ratio = total_unknown / total_responses if total_responses > 0 else 0
    print("\n" + "=" * 80)
    print(f"Overall: {total_unknown}/{total_responses} = {overall_ratio:.2%} unknown")
    print("=" * 80)

    # 问题诊断
    print("\n📊 Diagnosis:")
    if overall_ratio > 0.7:
        print("  ❌ CRITICAL: RAG system is returning 'I don't know' for most questions")
        print("     → Check retrieval quality (top_k, per_ctx_clip)")
        print("     → Check if questions are too specific")
    elif overall_ratio > 0.4:
        print("  ⚠️  WARNING: High unknown rate, may affect attack performance")
        print("     → Consider increasing top_k or per_ctx_clip")
    else:
        print("  ✅ OK: Unknown rate is acceptable")


def analyze_specific_doc(doc_id: str):
    """分析特定文档的详细信息"""

    if not os.path.exists(cache_path):
        print(f"❌ Cache file not found: {cache_path}")
        return

    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)

    print("=" * 80)
    print(f"Detailed Analysis for Document: {doc_id}")
    print("=" * 80)

    # 找到材料
    mat_key = None
    for key in cache.keys():
        if key.startswith(f"materials_{doc_id}_"):
            mat_key = key
            break

    if not mat_key:
        print(f"❌ No materials found for doc {doc_id}")
        return

    materials = cache[mat_key]
    questions = materials["questions"]

    # 找到 ground truth
    gt_key = f"ground_truths_{doc_id}_{len(questions)}"
    if gt_key not in cache:
        print(f"❌ No ground truths found for doc {doc_id}")
        return

    ground_truths = cache[gt_key]

    # 找到 RAG 响应
    rag_key = f"ia_rag_{doc_id}_nq{len(questions)}_defFalse"
    if rag_key not in cache:
        print(f"❌ No RAG responses found for doc {doc_id}")
        return

    rag_responses = cache[rag_key]["parsed_responses"]

    # 打印前 10 个问题和响应
    print(f"\nShowing first 10 questions and responses:\n")

    for i in range(min(10, len(questions))):
        gt = ground_truths[i]
        rag = rag_responses[i]

        # 检查是否匹配
        try:
            from attacks.ia_utils import parse_yes_no
            gt_parsed = parse_yes_no(gt)
            match = "✅" if rag == gt_parsed else "❌"
        except ImportError:
            # 如果无法导入，手动解析
            gt_clean = gt.lower().strip().rstrip('.')
            if gt_clean == "yes":
                gt_parsed = "yes"
            elif gt_clean == "no":
                gt_parsed = "no"
            elif "don't know" in gt_clean:
                gt_parsed = "unknown"
            else:
                gt_parsed = "unknown"
            match = "✅" if rag == gt_parsed else "❌"

        print(f"{i + 1}. Q: {questions[i][:80]}...")
        print(f"   GT: '{gt}' → parsed: '{gt_parsed}'")
        print(f"   RAG: '{rag}' {match}")
        print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze IA cache")
    parser.add_argument("--doc", type=str, help="Analyze specific document ID")

    args = parser.parse_args()

    if args.doc:
        analyze_specific_doc(args.doc)
    else:
        analyze_unknown_ratio()


if __name__ == "__main__":
    main()