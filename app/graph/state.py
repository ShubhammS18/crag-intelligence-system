from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict):
    #Input
    question : str 
    
    #After Retrieval
    docs : List[Document]
    
    #After eval_each_node 
    good_docs : List[Document]
    verdict : str 
    reason : str 
    
    # rewrite_query + web_search
    web_query : str              
    web_docs : List[Document]  
    
    # After refine node 
    strips : List[str]        
    kept_strips : List[str]      
    refined_context : str 
    
    answer : str


    
    