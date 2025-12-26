# src/mirabel/quick_eval_nfcorpus.py
from __future__ import annotations

import os
import json
from typing import List, Tuple

import numpy as np
import faiss
from beir.datasets.data_loader import GenericDataLoader
from FlagEmbedding import BGEM3FlagModel

from mirabel.threshold import mirabel_threshold, mirabel_decision_with_gap

def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2 normalize vectors so inner product approximates cosine similarity."""
    x = x.astype("float32")
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def load_index_and_mapping(index_dir: str = "index") -> Tuple[faiss.IndexFlatIP, dict]:
    """Load FAISS index and doc mapping (created by build_index_nfcorpus.py)."""
    index_path = os.path.join(index_dir, "nf.index")
    map_path = os.path.join(index_dir, "nf_docs.json")
    index = faiss.read_index(index_path)
    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)  # keys are stringified row indices
    return index, mapping

def eval_queries(
    queries: List[str],
    rho: float = 0.005,
    margin: float | None = 0.02,
    gap_min: float | None = 0.03,
    use_full_corpus: bool = True,
    topM_if_not_full: int = 200,
    use_gap: bool = True,
    use_precise_mu: bool = True,
) -> Tuple[int, int]:
    """
    Evaluate a list of queries and print per-query decision details.

    Parameters
    - queries: list of query strings to evaluate
    - rho: significance level for Gumbel threshold; smaller -> more conservative
    - margin: optional safety margin; if None, margin is disabled
      e.g., is_member requires s_max > tau + margin when margin is not None
    - gap_min: optional top-1 dominance requirement (s_max - s_2 >= gap_min);
      if None or use_gap=False, gap constraint is disabled
    - use_full_corpus: True to use the entire index for S_q (more stable threshold);
      False to use a limited candidate pool
    - topM_if_not_full: candidate pool size when use_full_corpus is False
    - use_gap: True to apply "Gumbel + margin + gap" combined decision;
      False to use a simpler rule (see comment block below)
    - use_precise_mu: True to use the precise μ_n formula; False uses the asymptotic approximation

    Returns
    - flagged: number of queries flagged as "member-like"
    - total: total number of evaluated queries

    Decision options:
    1) Combined (default):
       - s_max > tau + margin (if margin is not None), AND
       - (s_max - s_2) >= gap_min (if use_gap and gap_min is not None)
       via mirabel_decision_with_gap(...)

    2) Simpler (commented example below):
       - Only Gumbel: is_member_raw = (s_max > tau)
       - Gumbel + margin: is_member = (s_max > tau + margin)
    """
    index, mapping = load_index_and_mapping(index_dir="index")
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    total = 0
    flagged = 0

    # Decide candidate pool size
    topM_full = index.ntotal
    topM = topM_full if use_full_corpus else min(topM_if_not_full, topM_full)

    for q in queries:
        # Encode query
        q_out = model.encode([q], batch_size=1)
        q_vec = l2_normalize(q_out["dense_vecs"])

        # Search candidate pool
        D, I = index.search(q_vec, topM)
        scores = D[0].astype("float32")
        cand_ids = I[0].astype("int64")

        # ---------- Decision block ----------
        if use_gap and (gap_min is not None):
            # Combined rule: Gumbel + margin + gap
            is_member, tau, s_max, target_idx, s2, gap, res = mirabel_decision_with_gap(
                scores,
                rho=rho,
                margin=(0.0 if margin is None else margin),
                gap_min=gap_min,
                use_precise_mu=use_precise_mu,
            )
            target_doc_id = int(cand_ids[target_idx])
            is_member_str = f"{is_member}"
            print(f"\nQuery: {q}")
            print(
                f"  s_max={s_max:.4f}, tau={tau:.4f}, s_max-tau={s_max - tau:.4f}, "
                f"s2={s2:.4f}, gap={gap:.4f}, is_member={is_member_str}, target_doc_id={target_doc_id}"
            )
            print(f"  n={res.n}, mu_q={res.mu_q:.4f}, sigma_q={res.sigma_q:.4f}")
        else:
            # Simpler rule: pure Gumbel or Gumbel + margin
            res = mirabel_threshold(scores, rho=rho, use_precise_mu=use_precise_mu)
            s_max, tau = res.s_max, res.tau
            target_idx = res.target_idx
            target_doc_id = int(cand_ids[target_idx])

            is_member_raw = (s_max > tau)
            if margin is None:
                # Only Gumbel
                is_member = is_member_raw
                extra = "(rule=Gumbel)"
            else:
                # Gumbel + margin
                is_member = bool(s_max > (tau + margin))
                extra = f"(rule=Gumbel+margin, margin={margin})"

            print(f"\nQuery: {q}")
            print(
                f"  s_max={s_max:.4f}, tau={tau:.4f}, s_max-tau={s_max - tau:.4f}, "
                f"is_member={is_member} raw={is_member_raw} {extra}, target_doc_id={target_doc_id}"
            )
            print(f"  n={res.n}, mu_q={res.mu_q:.4f}, sigma_q={res.sigma_q:.4f}")

        # ---------- Summary update ----------
        total += 1
        flagged += int(is_member)

    print(
        f"\nSummary: {flagged}/{total} flagged as member "
        f"(rho={rho}, margin={margin}, gap_min={gap_min}, use_gap={use_gap}, use_full_corpus={use_full_corpus}, topM={topM})."
    )
    return flagged, total

def main():
    print("CWD:", os.getcwd())

    # A) Custom queries
    custom_queries = [
        "What are common symptoms related to medical conditions?",
        "What are common symptoms of asthma?",
        "How is insomnia diagnosed and treated?",
        "What causes eczema and how can it be managed?",
        "What are risk factors for cardiovascular disease?",
    ]
    # Example: combined rule (Gumbel + margin + gap)
    eval_queries(
        queries=custom_queries,
        rho=0.005,
        margin=0.02,
        gap_min=0.03,
        use_full_corpus=True,
        topM_if_not_full=200,
        use_gap=True,
        use_precise_mu=True,
    )

    # B) BEIR test queries
    data_root = os.path.join("data", "nfcorpus")
    _, queries_dict, _ = GenericDataLoader(data_folder=data_root).load(split="test")
    beir_queries = [queries_dict[qid] for qid in list(queries_dict.keys())[:20]]
    print("\n=== Evaluating BEIR test queries (first 20) ===")
    # Example: simpler rule — only Gumbel + margin, disable gap by setting use_gap=False
    eval_queries(
        queries=beir_queries,
        rho=0.005,
        margin=0.02,       # set to None to disable margin
        gap_min=0.03,      # ignored when use_gap=False
        use_full_corpus=True,
        topM_if_not_full=200,
        use_gap=True,     # switch off gap
        use_precise_mu=True,
    )

if __name__ == "__main__":
    main()