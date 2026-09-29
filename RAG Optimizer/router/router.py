"""Adaptive router — classify → registry → bandit → dispatch → (rerank)."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, replace

from config.settings import get_settings
from retrieval.base import BaseRetriever, QueryType, RetrievalResult
from retrieval.reranker.base_reranker import BaseReranker
from router.bandit import EpsilonGreedyBandit
from router.classifier import QueryClassifier
from router.registry import StrategyRegistry

log = logging.getLogger(__name__)


@dataclass
class RouterResult:
    retrieval_result: RetrievalResult
    selected_strategy: str
    query_type: QueryType
    was_exploration: bool
    classifier_latency_ms: float
    router_latency_ms: float
    reranked: bool


class AdaptiveRouter:
    def __init__(
        self,
        strategies: dict[str, BaseRetriever],
        reranker: BaseReranker | None = None,
        classifier: QueryClassifier | None = None,
        registry: StrategyRegistry | None = None,
        bandit: EpsilonGreedyBandit | None = None,
    ) -> None:
        self._strategies = strategies
        self._reranker = reranker
        self._classifier = classifier or QueryClassifier()
        self._registry = registry or StrategyRegistry()
        self._bandit = bandit or EpsilonGreedyBandit()
        self._settings = get_settings()

    async def warm_up(self) -> None:
        """Warm all strategies concurrently. Failed strategies are removed from the active set."""
        names = list(self._strategies.keys())
        results = await asyncio.gather(
            *[self._strategies[n].warm_up() for n in names],
            return_exceptions=True,
        )
        failed = [
            n for n, r in zip(names, results) if isinstance(r, Exception)
        ]
        for n in failed:
            exc = results[names.index(n)]
            log.warning("Strategy '%s' warm_up failed (removed from routing): %s", n, exc)
            del self._strategies[n]
        if not self._strategies:
            raise RuntimeError("All retrieval strategies failed warm_up — cannot serve requests")
        log.info("Warmed up strategies: %s", sorted(self._strategies.keys()))

    async def close_all(self) -> None:
        """Close retriever connections — called from FastAPI shutdown."""
        for strategy in self._strategies.values():
            if hasattr(strategy, "close"):
                try:
                    await strategy.close()
                except Exception:
                    pass

    async def route(
        self,
        query: str,
        top_k: int | None = None,
        *,
        force_strategy: str | None = None,
    ) -> RouterResult:
        """Route a query to the best strategy and return a fully annotated result.

        Args:
            query:          The user's raw query string.
            top_k:          Desired number of final result docs. Defaults to settings.rerank_top_n.
            force_strategy: If set, bypass classify + bandit and use this strategy directly.
                            Used by the benchmark harness and the /benchmark API endpoint.
        """
        router_start = time.perf_counter()
        top_k = top_k or self._settings.rerank_top_n

        # ── 1. Classify + select ───────────────────────────────────────
        classifier_start = time.perf_counter()
        if force_strategy:
            if force_strategy not in self._strategies:
                raise ValueError(
                    f"Unknown strategy '{force_strategy}'. "
                    f"Available: {sorted(self._strategies.keys())}"
                )
            query_type = QueryType.UNKNOWN
            selected = force_strategy
            was_exploration = False
        else:
            query_type = await self._classifier.classify(query)
            scores, counts = await self._registry.get_weights(query_type)
            selection = self._bandit.select(scores, counts)
            selected = selection.strategy
            was_exploration = selection.was_exploration

        classifier_latency_ms = (time.perf_counter() - classifier_start) * 1_000

        # ── 2. Retrieve ────────────────────────────────────────────────
        # Fetch extra candidates when a reranker is in the pipeline
        fetch_k = top_k * 3 if self._reranker else top_k
        retriever = self._strategies[selected]
        retrieval_result = await retriever.retrieve(query, top_k=fetch_k)

        # ── 3. Rerank ──────────────────────────────────────────────────
        reranked = False
        if self._reranker is not None:
            reranked_docs = await self._reranker.rerank(
                query, retrieval_result.docs, top_n=top_k
            )
            retrieval_result = replace(retrieval_result, docs=reranked_docs)
            reranked = True

        router_latency_ms = (time.perf_counter() - router_start) * 1_000

        return RouterResult(
            retrieval_result=retrieval_result,
            selected_strategy=selected,
            query_type=query_type,
            was_exploration=was_exploration,
            classifier_latency_ms=classifier_latency_ms,
            router_latency_ms=router_latency_ms,
            reranked=reranked,
        )
