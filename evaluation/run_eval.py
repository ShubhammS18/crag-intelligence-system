import os
import json
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerCorrectness, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import settings
from app.graph.graph import compiled_graph
from evaluation.dataset import EVAL_DATASET

EMPTY_STATE = {
    'docs': [], 'good_docs': [], 'verdict': '', 'reason': '',
    'web_query': '', 'web_docs': [], 'strips': [],
    'kept_strips': [], 'refined_context': '', 'answer': ''}


def _extract_score(val) -> float:
    """
    RAGAS returns scores as float, list, or numpy scalar depending on version.
    This handles all three cases safely.
    """
    if isinstance(val, list):
        # Filter out None and NaN values before averaging
        clean = [float(v) for v in val if v is not None and str(v) != 'nan']
        return round(sum(clean) / len(clean), 4) if clean else 0.0
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return 0.0


def run_crag_eval() -> dict:
    print('Running CRAG evaluation on', len(EVAL_DATASET), 'questions...')
    records = []

    for i, item in enumerate(EVAL_DATASET):
        result = compiled_graph.invoke({**EMPTY_STATE, 'question': item['question']})
        records.append({
            'user_input': item['question'],
            'response': result['answer'],
            'retrieved_contexts': result.get('kept_strips', []),
            'reference': item['reference']})
        print(f'[{i+1:02d}/{len(EVAL_DATASET)}] [{result.get("verdict","?")}] {item["question"][:55]}...')

    dataset = EvaluationDataset.from_list(records)

    evaluator_llm = LangchainLLMWrapper(
        ChatAnthropic(
            model=settings.haiku_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0))

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'))

    metrics = [Faithfulness(), AnswerCorrectness(), ContextPrecision(), ContextRecall()]

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings)

    scores = {
        'system': 'CRAG + Refine',
        'n_questions': len(EVAL_DATASET),
        'faithfulness':  _extract_score(results['faithfulness']),
        'answer_correctness': _extract_score(results['answer_correctness']),
        'context_precision': _extract_score(results['context_precision']),
        'context_recall': _extract_score(results['context_recall'])}

    os.makedirs('evaluation/results', exist_ok=True)
    with open('evaluation/results/crag_results.json', 'w') as f:
        json.dump(scores, f, indent=2)

    print('\nCRAG Scores:')
    for k, v in scores.items():
        if isinstance(v, float):
            print(f'{k}: {v:.4f}')
    print('Saved to evaluation/results/crag_results.json')
    return scores


if __name__ == '__main__':
    run_crag_eval()