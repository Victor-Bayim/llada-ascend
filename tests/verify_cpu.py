"""CPU checks of actual archived functions. No model weights or accelerator required."""
from pathlib import Path
import ast, json, csv, hashlib, platform, datetime
import torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
def load_functions(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    names={'_compute_logits_stats_chunked','get_num_transfer_tokens','_build_transfer_index'}
    nodes=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    nodes += [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    scope={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),scope)
    return scope
original=ROOT/'archive/original/analysis/llada_torch_npu/ceval_diffusion.py'
scope=load_functions(original)
checks=[]
class ToyModel:
    def __init__(self,h,w,b):
        self.h=h
        self.lm_head=torch.nn.Linear(w.shape[1],w.shape[0],bias=b is not None)
        self.lm_head.weight.data.copy_(w)
        if b is not None:self.lm_head.bias.data.copy_(b)
    def model(self,**kwargs):return (self.h,)
for seed in [7,19,41]:
    torch.manual_seed(seed)
    h=torch.randn(2,7,13); w=torch.randn(37,13)
    ids=torch.zeros((2,7),dtype=torch.long)
    for use_bias in [False,True]:
        b=torch.randn(37) if use_bias else None
        model=ToyModel(h,w,b)
        logits=torch.nn.functional.linear(h,w,b)
        expected=logits.argmax(-1);prob=logits.softmax(-1).gather(-1,expected.unsqueeze(-1)).squeeze(-1)
        for chunk in [1,8,37]:
            for masked in [False,True]:
                mask=torch.rand(2,7)>.5 if masked else None
                actual,confidence=scope['_compute_logits_stats_chunked'](model,ids,mask,None,0.0,chunk,False)
                chosen=mask if masked else torch.ones_like(ids,dtype=torch.bool)
                assert torch.equal(actual[chosen],expected[chosen])
                torch.testing.assert_close(confidence[chosen],prob[chosen],rtol=3e-5,atol=3e-6)
                if masked: assert torch.equal(actual[~mask],ids[~mask])
                checks.append({'check':'chunked logits vs full projection','seed':seed,'bias':use_bias,'chunk':chunk,'masked':masked,'passed':True})
mask=torch.tensor([[1,1,1,1,1,0],[1,1,0,0,0,0]],dtype=torch.bool)
counts=scope['get_num_transfer_tokens'](mask,4)
assert torch.equal(counts.sum(1),mask.sum(1)); checks.append({'check':'transfer schedule preserves mask counts','passed':True})
counterexample=None
for seed in range(100):
    torch.manual_seed(seed);confidence=torch.randn(3,1000);k=torch.tensor([[100],[7],[0]])
    ref=scope['_build_transfer_index'](confidence,k,0,False,None)
    batched=scope['_build_transfer_index'](confidence,k,0,True,None)
    if not torch.equal(ref,batched):counterexample={'seed':seed,'k':[100,7,0]};break
assert counterexample is not None
patched=load_functions(ROOT/'analysis/llada_torch_npu/ceval_diffusion.py')
for seed in range(100):
    torch.manual_seed(seed);confidence=torch.randn(3,1000);k=torch.tensor([[100],[7],[0]])
    ref=scope['_build_transfer_index'](confidence,k,0,False,None)
    got=patched['_build_transfer_index'](confidence,k,0,True,None)
    assert torch.equal(ref,got)
checks.append({'check':'maintenance patch: batched top-k matches row loop for 100 seeds','passed':True})
manifest=json.loads((ROOT/'SOURCE-MANIFEST.json').read_text())
for item in manifest['files']:
    assert hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()==item['sha256']
rows=list(csv.DictReader((ROOT/'data/reported-latency.csv').open()))
assert len(rows)==14
for row in rows:assert float(row['latency_seconds'])>0 and row['input_tokens']==row['output_tokens']
checks.append({'check':'14 report rows and original source hashes','passed':True})
result={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'platform':platform.system()+' '+platform.machine(),'torch':torch.__version__,'scope':'CPU functional checks; no NPU latency or model-quality remeasurement','passed_checks':len(checks),'checks':checks,'original_issue':{'description':'unsorted batched top-k can select wrong smaller-k subset','counterexample':counterexample,'status':'archive preserved; maintained entry point includes sorted top-k fix'},'npu_benchmark_rerun':False}
(ROOT/'validation').mkdir(exist_ok=True)
(ROOT/'validation/current-cpu-checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['platform','torch','passed_checks','scope','original_issue']}))
