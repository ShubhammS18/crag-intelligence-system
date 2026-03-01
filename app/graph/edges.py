from app.graph.state import GraphState


def route_after_eval(state: GraphState) -> str:
    """
    CRAG routing decision after eval_each_doc.
    CORRECT  -> refine directly (good internal docs exist)
    AMBIGUOUS / INCORRECT -> rewrite_query first to fetch web context
    """
    if state['verdict'] == 'CORRECT':
        return 'refine'
    return 'rewrite_query'
