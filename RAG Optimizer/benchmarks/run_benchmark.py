"""Benchmark CLI — orchestrates BEIR evaluation, prints comparison table, enforces CI gate.

Usage examples:
    python benchmarks/run_benchmark.py --dataset nfcorpus --max-queries 50
    python benchmarks/run_benchmark.py --dataset nfcorpus --strategies dense hybrid
    python benchmarks/run_benchmark.py --dataset nfcorpus --ci --baseline-file benchmarks/baselines/nfcorpus.json
    python benchmarks/run_benchmark.py --dataset nfcorpus --save-baseline

Exit codes: 0 = pass, 1 = regression detected, 2 = infrastructure error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from benchmarks.beir_eval import BEIREvaluator, MetricsResult, check_regression
from retrieval.strategies.bm25 import BM25Retriever
from retrieval.strategies.dense import DenseRetriever
from retrieval.strategies.hybrid import HybridRetriever
from retrieval.strategies.hyde import HyDERetriever
from retrieval.strategies.multi_query import MultiQueryRetriever
from retrieval.strategies.parent_child import ParentChildRetriever
from router.registry import StrategyRegistry
from router.router import AdaptiveRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_RESULTS_DIR = Path("benchmarks/results")
_BASELINES_DIR = Path("benchmarks/baselines")
_ALL_STRATEGIES = ["dense", "bm25", "hybrid", "hyde", "multi_query", "parent_child"]


# ---------------------------------------------------------------------------
# Router construction
# ---------------------------------------------------------------------------

async def build_router() -> tuple[AdaptiveRouter, list[str]]:
    """Build and warm up all strategies. Returns (router, list_of_available_strategies)."""
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    child_dense = DenseRetriever(collection="rag_dense_children")

    candidates = {
        "dense": dense,
        "bm25": bm25,
        "hybrid": HybridRetriever(dense=dense, bm25=bm25),
        "hyde": HyDERetriever(dense=dense),
        "multi_query": MultiQueryRetriever(dense=dense),
        "parent_child": ParentChildRetriever(dense=child_dense),
    }

    available: dict = {}
    for name, retriever in candidates.items():
        try:
            await retriever.warm_up()
            available[name] = retriever
            log.info("  ✓ %s", name)
        except Exception as exc:
            log.warning("  ✗ %s — warm_up failed: %s", name, exc)

    if not available:
        log.error("No strategies could be warmed up — check that Qdrant and ES are running.")
        sys.exit(2)

    registry = StrategyRegistry()  # no DB pool — fine for forced-strategy benchmarking
    router = AdaptiveRouter(strategies=available, registry=registry)
    return router, list(available.keys())


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

def _print_table(
    dataset: str,
    results: dict[str, MetricsResult],
    baseline: dict[str, MetricsResult] | None = None,
) -> None:
    best = max(results, key=lambda s: results[s].primary()) if results else None

    col_w = {"strategy": 14, "ndcg": 10, "mrr": 10, "recall": 11, "ms": 6}
    h_top = "╔" + "═" * col_w["strategy"] + "╦" + "═" * col_w["ndcg"] + "╦" + "═" * col_w["mrr"] + "╦" + "═" * col_w["recall"] + "╦" + "═" * col_w["ms"] + "╗"
    h_sep = "╠" + "═" * col_w["strategy"] + "╬" + "═" * col_w["ndcg"] + "╬" + "═" * col_w["mrr"] + "╬" + "═" * col_w["recall"] + "╬" + "═" * col_w["ms"] + "╣"
    h_bot = "╚" + "═" * col_w["strategy"] + "╩" + "═" * col_w["ndcg"] + "╩" + "═" * col_w["mrr"] + "╩" + "═" * col_w["recall"] + "╩" + "═" * col_w["ms"] + "╝"

    def _row(strategy: str, m: MetricsResult) -> str:
        marker = " ↑" if strategy == best else "  "
        ndcg = f"{m.primary():.3f}{marker}"
        mrr = f"{m.metrics.get('mrr', 0.0):.3f}"
        recall = f"{m.metrics.get('recall@100', m.metrics.get('recall@10', 0.0)):.3f}"
        ms = f"{m.avg_latency_ms:.0f}"
        return (
            f"║ {strategy:<{col_w['strategy']-2}} ║"
            f" {ndcg:>{col_w['ndcg']-2}} ║"
            f" {mrr:>{col_w['mrr']-2}} ║"
            f" {recall:>{col_w['recall']-2}} ║"
            f" {ms:>{col_w['ms']-1}}║"
        )

    print(f"\n╔{'═' * (sum(col_w.values()) + len(col_w) - 1)}╗")
    print(f"║  BEIR Evaluation — {dataset:<{sum(col_w.values()) + len(col_w) - 22}}║")
    print(h_top)
    print(f"║ {'Strategy':<{col_w['strategy']-2}} ║ {'nDCG@10':>{col_w['ndcg']-2}} ║ {'MRR':>{col_w['mrr']-2}} ║ {'R@100':>{col_w['recall']-2}} ║ {'P50ms':>{col_w['ms']-1}}║")
    print(h_sep)

    sorted_results = sorted(results.items(), key=lambda x: x[1].primary(), reverse=True)
    for strategy, m in sorted_results:
        print(_row(strategy, m))

    print(h_bot)

    if best and baseline and best in baseline:
        delta = results[best].primary() - baseline[best].primary()
        sign = "+" if delta >= 0 else ""
        print(f"Best: {best} ({sign}{delta:.3f} vs baseline {best})")
    elif best:
        print(f"Best: {best} ({results[best].primary():.3f} nDCG@10)")


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------

def save_baseline(dataset: str, results: dict[str, MetricsResult]) -> Path:
    _BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    path = _BASELINES_DIR / f"{dataset}.json"
    payload = {s: m.to_dict() for s, m in results.items()}
    path.write_text(json.dumps(payload, indent=2))
    log.info("Baseline saved → %s", path)
    return path


def load_baseline(path: str | Path) -> dict[str, MetricsResult]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return {s: MetricsResult.from_dict(v) for s, v in data.items()}


def save_report(dataset: str, results: dict[str, MetricsResult]) -> Path:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _RESULTS_DIR / f"{dataset}_{ts}.json"
    best = max(results, key=lambda s: results[s].primary()) if results else ""
    strategies_data = {s: m.to_dict() for s, m in results.items()}
    strategies_ndcg = {s: m.primary() for s, m in results.items()}
    baseline_strategy = "bm25" if "bm25" in results else (best or "")
    improvement = (
        results[best].primary() - results[baseline_strategy].primary()
        if best and baseline_strategy and best != baseline_strategy
        else 0.0
    )
    payload = {
        "dataset": dataset,
        "timestamp": datetime.now().isoformat(),
        "strategies": {s: {**m.metrics, "avg_latency_ms": m.avg_latency_ms} for s, m in results.items()},
        "best_strategy": best,
        "baseline_strategy": baseline_strategy,
        "improvement_over_baseline": round(improvement, 4),
    }
    path.write_text(json.dumps(payload, indent=2))
    log.info("Report saved → %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> int:
    log.info("Building router …")
    router, available = await build_router()

    target_strategies = args.strategies or available
    invalid = [s for s in target_strategies if s not in available]
    if invalid:
        log.warning("Strategies not available (infra may be down): %s", invalid)
        target_strategies = [s for s in target_strategies if s in available]

    if not target_strategies:
        log.error("No strategies available.")
        return 2

    evaluator = BEIREvaluator(router=router)
    log.info("Evaluating %s on %s (max_queries=%s) …", target_strategies, args.dataset, args.max_queries)

    results = await evaluator.evaluate(
        dataset=args.dataset,
        strategies=target_strategies,
        k_values=[5, 10, 100],
        max_queries=args.max_queries,
    )

    baseline = load_baseline(args.baseline_file) if args.baseline_file else {}
    _print_table(args.dataset, results, baseline or None)
    save_report(args.dataset, results)

    if args.save_baseline:
        save_baseline(args.dataset, results)

    if args.ci:
        if not baseline:
            log.warning("--ci set but no baseline loaded — saving current run as baseline.")
            save_baseline(args.dataset, results)
            print("CI gate: PASSED (first run — baseline saved)")
            return 0
        passed, msg = check_regression(results, baseline)
        print(f"CI gate: {'PASSED' if passed else 'FAILED'}")
        if not passed:
            print(msg)
            return 1

    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG benchmark runner with CI regression gate")
    p.add_argument("--dataset", default="nfcorpus")
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument(
        "--strategies", nargs="+", default=None, choices=_ALL_STRATEGIES,
        help="Strategies to evaluate. Default: all available.",
    )
    p.add_argument("--ci", action="store_true", help="Enable CI regression gate (exit 1 on regression)")
    p.add_argument("--baseline-file", default=None, help="Path to baseline JSON for CI comparison")
    p.add_argument("--save-baseline", action="store_true", help="Write results as new baseline")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(asyncio.run(_run(args)))
