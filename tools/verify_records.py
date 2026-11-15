from pathlib import Path
import json, hashlib, csv
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'SOURCE-MANIFEST.json').read_text())
for item in manifest['files']:
    assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256'], item['path']
rows=list(csv.DictReader((root/'data/reported-latency.csv').open()))
assert len(rows)==14
assert {(int(r['batch']),int(r['input_tokens'])) for r in rows}=={(b,n) for b in [1,16] for n in [32,64,128,256,512,1024,2048]}
for row in rows:
    assert row['input_tokens']==row['output_tokens'] and row['steps']=='2'
    assert float(row['latency_seconds'])>0
    assert row['source_type']=='technical-report transcription'
print(json.dumps({'source_hashes':len(manifest['files']),'report_rows':len(rows),'status':'passed','scope':'source integrity and report-table checks; not an NPU rerun'}))
