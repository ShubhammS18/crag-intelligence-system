from unittest.mock import patch, MagicMock
from langchain_core.documents import Document


def make_doc(text: str) -> Document:
    return Document(page_content=text, metadata={'source': 'test'})


# eval_each_doc tests
class TestEvalEachDoc:

    def test_verdict_correct_when_high_score(self):
        from app.graph.nodes import eval_each_doc
        state = {
            'question': 'What are UAE visa rules?',
            'docs': [make_doc('UAE visa rules for AI engineers with 2+ years')]
        }
        mock_out = MagicMock(score=0.85, reason='highly relevant')
        with patch('app.graph.nodes.doc_eval_chain') as mock:
            mock.invoke.return_value = mock_out
            result = eval_each_doc(state)
        assert result['verdict'] == 'CORRECT'
        assert len(result['good_docs']) == 1

    def test_verdict_incorrect_when_all_low_score(self):
        from app.graph.nodes import eval_each_doc
        state = {
            'question': 'What is the Bitcoin price?',
            'docs': [make_doc('DIFC tax rate is 0% for startups under 3M AED')]
        }
        mock_out = MagicMock(score=0.05, reason='irrelevant')
        with patch('app.graph.nodes.doc_eval_chain') as mock:
            mock.invoke.return_value = mock_out
            result = eval_each_doc(state)
        assert result['verdict'] == 'INCORRECT'
        assert result['good_docs'] == []

    def test_verdict_ambiguous_when_mixed_scores(self):
        from app.graph.nodes import eval_each_doc
        state = {
            'question': 'What is the tax rate and current Bitcoin price?',
            'docs': [
                make_doc('DIFC tax rate is 0%'),
                make_doc('Unrelated content about weather'),
            ]
        }
        mock_outs = [
            MagicMock(score=0.55, reason='partially relevant'),
            MagicMock(score=0.10, reason='irrelevant'),
        ]
        with patch('app.graph.nodes.doc_eval_chain') as mock:
            mock.invoke.side_effect = mock_outs
            result = eval_each_doc(state)
        assert result['verdict'] == 'AMBIGUOUS'
        assert len(result['good_docs']) == 1  # only the 0.55 doc kept


# refine tests 
class TestRefine:

    def test_refine_drops_irrelevant_sentences(self):
        from app.graph.nodes import refine
        doc = make_doc(
            'Engineers qualify for Golden Visa. The sky is blue today. Apply via DIFC portal.'
        )
        state = {
            'question': 'Who qualifies for the Golden Visa?',
            'verdict': 'CORRECT',
            'good_docs': [doc],
            'web_docs':  []
        }
        keep_values = [
            MagicMock(keep=True),   # Engineers qualify
            MagicMock(keep=False),  # sky is blue — dropped
            MagicMock(keep=True),   # Apply via DIFC portal
        ]
        with patch('app.graph.nodes.filter_chain') as mock:
            mock.invoke.side_effect = keep_values
            result = refine(state)
        assert len(result['kept_strips']) == 2
        assert 'sky is blue' not in result['refined_context']

    def test_refine_uses_web_docs_when_incorrect(self):
        from app.graph.nodes import refine
        web_doc = make_doc('Bitcoin is currently trading at $95000.')
        state = {
            'question': 'What is Bitcoin price?',
            'verdict': 'INCORRECT',
            'good_docs': [],
            'web_docs': [web_doc]
        }
        with patch('app.graph.nodes.filter_chain') as mock:
            mock.invoke.return_value = MagicMock(keep=True)
            result = refine(state)
        assert len(result['kept_strips']) > 0
        assert 'Bitcoin' in result['refined_context']

