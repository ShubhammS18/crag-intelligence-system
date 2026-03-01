from langgraph.graph import StateGraph, START, END
from app.graph.state import GraphState
from app.graph.nodes import (retrieve, eval_each_doc, rewrite_query,
                            web_search, refine, generate)
from app.graph.edges import route_after_eval


def build_graph():
    g = StateGraph(GraphState)

    # nodes 
    g.add_node('retrieve', retrieve)
    g.add_node('eval_each_doc', eval_each_doc)
    g.add_node('rewrite_query', rewrite_query)
    g.add_node('web_search', web_search)
    g.add_node('refine', refine)
    g.add_node('generate', generate)

    # edges 
    g.add_edge(START, 'retrieve')
    g.add_edge('retrieve', 'eval_each_doc')

    # CRAG conditional routing 
    # CORRECT  -> refine
    # AMBIGUOUS/INCORRECT -> rewrite_query -> web_search -> refine
    g.add_conditional_edges('eval_each_doc',route_after_eval,
        {'refine': 'refine', 'rewrite_query': 'rewrite_query'}
    )

    # Non-correct path 
    g.add_edge('rewrite_query', 'web_search')
    g.add_edge('web_search', 'refine')

    g.add_edge('refine', 'generate')
    g.add_edge('generate', END)

    return g.compile()

compiled_graph = build_graph()
