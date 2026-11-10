#!/usr/bin/env python3
"""Precompute matched saved replays; references never enter extraction requests."""
import argparse,copy,gzip,hashlib,json,random,sys,tarfile,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from adaptgym.extraction_eval import request_for,score_row,summarize
from adaptgym.intent_contract import content_hash
from adaptgym.openai_session_test import ResponsesClient,EXTRACT_SCHEMA
from debug_session_scenarios import load_env_file

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)

def public_fingerprints(value):
    # Algorithm-labelled fingerprints retain provenance without looking like bare credentials.
    if isinstance(value,dict):
        return {k:('sha256:'+v if k.endswith('_sha256') and isinstance(v,str) and len(v)==64 else public_fingerprints(v)) for k,v in value.items()}
    if isinstance(value,list):return [public_fingerprints(v) for v in value]
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--env-file',type=Path,default=ROOT/'.env');p.add_argument('--workers',type=int,default=5);a=p.parse_args();load_env_file(a.env_file)
    data=a.run/'app/data';records=json.loads(gzip.decompress((data/'test-records.json.gz').read_bytes()));rng=random.Random(20261110)
    selected=[]
    for vertical in ('footwear','job_search'):selected+=rng.sample(sorted(r['id'] for r in records if r['vertical']==vertical),5)
    records={r['id']:r for r in records if r['id'] in selected};rows={sid:[] for sid in selected}
    for r in map(json.loads,gzip.open(data/'test.jsonl.gz','rt')):
        if r['sample_id'] in rows:rows[r['sample_id']].append(r)
    frozen={}
    for name in ('baseline','lora'):
        with tarfile.open(a.run/(name+'-snapshot.tar.gz')) as tar:
            frozen[name]={(r['sample_id'],r['mode'],r['id']):r for r in map(json.loads,gzip.decompress(tar.extractfile('predictions.jsonl.gz').read()).decode().splitlines())}
    messages={}
    source=a.run.parent/'qwen1600-jev-20261109/gateway-stream/conversations.jsonl.gz'
    for line in gzip.open(source,'rt'):
        item=json.loads(line)
        if item['id'] in rows:messages[item['id']]=item['payload']
    private=a.run/'ten-trace-luna';private.mkdir(exist_ok=True)
    def luna(sid,mode):
        path=private/(sid+'-'+mode+'.json');saved=json.loads(path.read_text()) if path.exists() else {'rows':[]}
        client=ResponsesClient('gpt-6-luna',max_calls=len(rows[sid]),max_output_tokens=4096,timeout=300)
        result=saved['rows'];previous=None;client.calls=len(result)
        for old in result:
            if old['metrics']['score']['valid']:previous=old['prediction']
        for r in rows[sid][len(result):]:
            prompt,payload=request_for(r,previous,mode=='rollout');prediction=None;error=None;started=time.monotonic()
            try:prediction=client.json(prompt,payload,EXTRACT_SCHEMA,purpose='frozen_extraction')
            except Exception as exc:error=type(exc).__name__
            metrics=score_row(prediction,r,records[sid],error)
            item={'id':r['id'],'sample_id':sid,'vertical':records[sid]['vertical'],'mode':mode,'prediction':prediction,'metrics':metrics,
                  'input_sha256':content_hash({'prompt':prompt,'payload':payload}),'elapsed_seconds':time.monotonic()-started,'trace':client.traces[-1] if client.traces else None}
            result.append(item)
            if metrics['score']['valid']:previous=prediction
            save(path,{'completed':len(result)==len(rows[sid]),'rows':result})
            print(json.dumps({'sample':sid,'mode':mode,'turns':len(result),'valid':metrics['score']['valid']}),flush=True)
        return sid,mode,result
    luna_results={}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(luna,sid,mode) for sid in selected for mode in ('teacher_forced','rollout')]):
            sid,mode,result=f.result();luna_results[sid,mode]=result
    index=[];totals={};counts={'footwear':0,'job_search':0}
    for sid in selected:
        record=records[sid];vertical=record['vertical'];counts[vertical]+=1;alias=('sft-shoes' if vertical=='footwear' else 'sft-jobs')+'-'+str(counts[vertical]);plan=record['plan'];journal=messages[sid]
        sample={'sample_id':alias,'plan':plan,'model':'Qwen/Qwen3.8-27B','generation_only':True,
                'result':{'status':'completed','conversation':journal['messages'],'checkpoints':[{'turn':int(r['id'].rsplit(':',1)[1]),'prediction':None,'score':{'valid':False,'not_run':True},'reference_target':json.loads(r['messages'][2]['content']),
                    'verification':{'status':'completed','accepted':True,'method':'automated binary meaning-preservation gate'}} for r in rows[sid]]}}
        sample_sha=content_hash(sample);models={}
        for key,label in (('luna','Luna'),('baseline','Qwen 4B zero-shot'),('lora','Qwen 4B fine-tuned')):
            models[key]={}
            for mode in ('teacher_forced','rollout'):
                items=luna_results[sid,mode] if key=='luna' else [frozen[key][sid,mode,r['id']] for r in rows[sid]]
                if mode=='teacher_forced':
                    assert all(x['input_sha256']==frozen['baseline'][sid,mode,x['id']]['input_sha256'] for x in items),'input parity failed'
                checkpoints=[];previous=None
                for r,item in zip(rows[sid],items):
                    turn=int(r['id'].rsplit(':',1)[1]);profile=item['prediction'];metrics=item['metrics'];ref=json.loads(r['messages'][2]['content']);step=plan['exchanges'][turn-1]
                    prefix=json.loads(r['messages'][1]['content'])['visible_conversation'];reply=next(m for m in journal['messages'] if m['id']==step['assistant']['message_id'])
                    cp={'turn':turn,'profile':profile,'reference_profile':ref,'score':metrics['score'],'verification':sample['result']['checkpoints'][turn-1]['verification'],
                        'previous_prediction':json.loads(r['messages'][1]['content'])['previous_prediction'] if mode=='teacher_forced' else previous,
                        'user_message':prefix[-1]['text'],'reply':reply['text'],'messages':prefix+[reply],'done':turn==len(rows[sid]),
                        'products':[x for x in plan['catalog']['products'] if x['id'] in step['assistant']['product_ids']],
                        'evaluation_metrics':metrics,'input_sha256':item['input_sha256'],
                        'metrics':{'turns':turn,'total_turns':len(rows[sid]),'extraction_ms':round(item['elapsed_seconds']*1000),
                            'value_precision':metrics['value_tp']/(metrics['value_tp']+metrics['value_fp']) if metrics['value_tp']+metrics['value_fp'] else None,
                            'value_recall':metrics['value_tp']/(metrics['value_tp']+metrics['value_fn']) if metrics['value_tp']+metrics['value_fn'] else None,
                            'attributes':len({c['attribute'] for c in (profile or {}).get('claims',[])}),'targets':len((profile or {}).get('target_products',[])),'selected':len((profile or {}).get('selected_products',[])),'item_preferences':len((profile or {}).get('preferences',[])),
                            'preferences_correct':metrics['score'].get('preferences_correct',False),'attempts':turn,'valid_outputs':sum(x['metrics']['score']['valid'] for x in items[:turn])}}
                    checkpoints.append(cp)
                    if metrics['score']['valid']:previous=profile
                models[key][mode]={'sample_id':alias,'vertical':vertical,'model':items[0]['trace']['model'],'label':label,'completed':True,'state_mode':mode,
                    'sample_sha256':sample_sha,'extractor_prompt':rows[sid][0]['messages'][0]['content'],'prompt_sha256':hashlib.sha256(rows[sid][0]['messages'][0]['content'].encode()).hexdigest(),
                    'checkpoints':checkpoints,'summary':summarize(items),
                    'api_traces':[{k:v for k,v in (x.get('trace') or {}).items() if k in ('model','status','wall_seconds','usage','finish_reason','error_type')} for x in items]}
                totals.setdefault(key,{}).setdefault(mode,[]).extend(items)
        bundle={'id':alias,'vertical':vertical,'sample':sample,'models':models}
        save(a.output/(alias+'.json'),public_fingerprints(bundle));index.append({'id':alias,'vertical':vertical,'file':alias+'.json','label':('Shoes' if vertical=='footwear' else 'Jobs')+' '+str(counts[vertical])+' · '+str(len(rows[sid]))+' turns','turns':len(rows[sid])})
    save(a.output/'index.json',{'selection':{'method':'seeded random; five per vertical; test only','seed':20261110},'models':[{'id':k,'label':v} for k,v in [('luna','Luna'),('baseline','Qwen 4B zero-shot'),('lora','Qwen 4B fine-tuned')]],'cases':index,
         'summary':{key:{mode:summarize(items) for mode,items in modes.items()} for key,modes in totals.items()}})
    print('COMPARISON_BUNDLE_COMPLETE',flush=True)
if __name__=='__main__':main()
