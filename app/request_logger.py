"""
Request logger for the CRAG Intelligence System.

Writes one JSONL entry per /ask request to request_log.jsonl.
This file is the foundation for computing production metrics:
  - Failure rate:       count success=false / total
  - Citation coverage:  count kept_strips > 0 / total
  - Verdict distribution: group by verdict field
  - Avg latency:        mean of latency_ms
  - Avg cost:           mean of estimated_cost_usd
  - Cost per verdict:   group by verdict, mean of estimated_cost_usd

"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

LOG_FILE = Path('request_log.jsonl')
logger   = logging.getLogger('crag-logger')


def log_request(
    question:  str,
    verdict: str,
    latency_ms: int,
    kept_strips_count: int,
    estimated_cost_usd: float,
    prompt_version: str,
    error: Exception = None,) -> None:
    
    """
    Append one structured log entry to request_log.jsonl.
    Never raises — logging failures must not crash the API.
    """
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'question_preview': question[:80],   # truncated for privacy
        'verdict':  verdict,
        'latency_ms':  latency_ms,
        'kept_strips_count':   kept_strips_count,
        'estimated_cost_usd':  estimated_cost_usd,
        'prompt_version':  prompt_version,
        'success':  error is None,
        'error':  str(error) if error else None}
    try:
        with LOG_FILE.open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
    except Exception as log_error:
        # Log to console but never crash the API due to logging failure
        logger.error(f'Failed to write request log: {log_error}')


def compute_metrics() -> dict:
    """
    Read request_log.jsonl and compute summary metrics.
    Returns a dict suitable for a /metrics endpoint or README reporting.
    """
    if not LOG_FILE.exists():
        return {'error': 'No requests logged yet'}

    entries = []
    with LOG_FILE.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not entries:
        return {'error': 'Log file is empty'}

    total = len(entries)
    successes = [e for e in entries if e.get('success')]
    failures = [e for e in entries if not e.get('success')]
    latencies = [e['latency_ms'] for e in successes if 'latency_ms' in e]
    costs = [e['estimated_cost_usd'] for e in successes if 'estimated_cost_usd' in e]
    cited = [e for e in successes if e.get('kept_strips_count', 0) > 0]

    # Verdict distribution
    verdicts = {}
    for e in entries:
        v = e.get('verdict', 'UNKNOWN')
        verdicts[v] = verdicts.get(v, 0) + 1

    # Latency percentiles
    sorted_lat = sorted(latencies)
    def pct(lst, p):
        if not lst: return 0
        idx = int(len(lst) * p / 100)
        return lst[min(idx, len(lst)-1)]

    return {
        'total_requests':  total,
        'success_rate':    round(len(successes) / total, 4) if total else 0,
        'failure_rate':    round(len(failures)  / total, 4) if total else 0,
        'citation_coverage':  round(len(cited) / len(successes), 4) if successes else 0,
        'latency_p50_ms':  pct(sorted_lat, 50),
        'latency_p95_ms':  pct(sorted_lat, 95),
        'avg_latency_ms':  round(sum(latencies) / len(latencies), 1) if latencies else 0,
        'avg_cost_usd':    round(sum(costs) / len(costs), 6) if costs else 0,
        'total_cost_usd':  round(sum(costs), 6),
        'verdict_distribution': verdicts,
        'prompt_version':  entries[-1].get('prompt_version', 'unknown')}
