from fastapi.testclient import TestClient
from unittest.mock import patch


def get_client():
    from app.api import app
    return TestClient(app)


def test_health_check():
    client = get_client()
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_ask_returns_structured_response():
    client = get_client()
    mock_result = {
        'answer':  'The corporate tax rate is 0% for startups under 3M AED.',
        'verdict':  'CORRECT',
        'reason':  'At least one chunk scored > 0.7.',
        'kept_strips':  ['The corporate tax rate is 0% for startups under 3M AED.'],
        'docs':            [],
        'good_docs':       [],
        'web_query':       '',
        'web_docs':        [],
        'strips':          [],
        'refined_context': 'The corporate tax rate is 0% for startups under 3M AED.',
        'question':  'What is the tax rate?',
        'latency_ms':   1234,
        'estimated_cost_usd': 0.005}
    
    
    with patch('app.api.compiled_graph') as mock_graph:
        mock_graph.invoke.return_value = mock_result
        response = client.post('/ask', json={'question': 'What is the tax rate?'})
    assert response.status_code == 200
    data = response.json()
    assert 'answer'  in data
    assert 'verdict' in data
    assert data['verdict'] == 'CORRECT'
    assert 'latency_ms'  in data
    assert 'estimated_cost_usd' in data



def test_ask_rejects_empty_question():
    client = get_client()
    response = client.post('/ask', json={'question': '   '})
    assert response.status_code == 400
