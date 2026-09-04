import json
d = json.load(open('evaluation/results/results_baseline_rag_20260902_224315.json'))
print('Total queries:', len(d))
for r in d[:3]:
    m = r.get("metrics") or {}
    print(f"id={r['id']} | mrr={m.get('mrr')} | recall@12={m.get('recall@12')} | critic_passed={r.get('critic_passed')} | retries={r.get('retry_count')}")
    print(f"   relevant_ids ({len(r.get('relevant_chunk_ids') or [])}): {(r.get('relevant_chunk_ids') or [])[:3]}")
    print(f"   retrieved_ids ({len(r.get('retrieved_chunk_ids') or [])}): {(r.get('retrieved_chunk_ids') or [])[:5]}")
    print()
