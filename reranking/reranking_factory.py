from .base import BaseReranker, RerankingStrategy
from .original_order import OriginalOrderReranker
from .u_shape_reorder import UShapeReranker


_REGISTRY: dict[RerankingStrategy, type[BaseReranker]] = {
    RerankingStrategy.ORIGINAL_ORDER: OriginalOrderReranker,
    RerankingStrategy.U_SHAPE_REORDER: UShapeReranker,
}


def create_reranker(strategy: RerankingStrategy) -> BaseReranker:
    reranker_cls = _REGISTRY.get(strategy)
    if reranker_cls is None:
        raise ValueError(
            f"Unknown reranking strategy: '{strategy}'. "
            f"Available strategies: {[s.value for s in RerankingStrategy]}"
        )
    return reranker_cls()