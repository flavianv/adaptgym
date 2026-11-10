"""Paired extraction metrics; current-turn reference labels never enter requests."""
import copy,json
from adaptgym.openai_session_test import score_extraction,value_signatures
from adaptgym.verticals import VerticalSchema

STATE_FIELDS=('active_task_correct','selected_products_correct','target_products_correct',
              'preferences_correct','outcome_correct','search_constraints_correct')

def request_for(row,previous=None,rollout=False):
    payload=json.loads(row['messages'][1]['content'])
    if rollout:payload['previous_prediction']=copy.deepcopy(previous)
    return row['messages'][0]['content'],payload

def score_row(prediction,row,record,error=None):
    reference=json.loads(row['messages'][2]['content']);payload=json.loads(row['messages'][1]['content'])
    step=record['plan']['exchanges'][int(row['id'].rsplit(':',1)[1])-1]
    expected=value_signatures(reference['claims'])
    score=({'valid':False,'error_type':error or 'NoPrediction'} if prediction is None else
        score_extraction(prediction,step,VerticalSchema(record['plan']['vertical_schema']),
            payload['visible_conversation'],payload['catalog']))
    valid=score['valid'];actual=value_signatures(prediction['claims']) if valid else set()
    return {'score':score,'value_tp':len(expected&actual),'value_fp':len(actual-expected),'value_fn':len(expected-actual),
        'complete_value_coverage':valid and expected<=actual,'exact_values':valid and expected==actual,
        'strict_exact_state':valid and score.get('strict_exact_state',False),
        'missing_values':[json.loads(x) for x in sorted(expected-actual)],
        'extra_values':[json.loads(x) for x in sorted(actual-expected)]}

def summarize(rows):
    n=len(rows);valid=sum(r['metrics']['score']['valid'] for r in rows)
    tp=sum(r['metrics']['value_tp'] for r in rows);fp=sum(r['metrics']['value_fp'] for r in rows);fn=sum(r['metrics']['value_fn'] for r in rows)
    result={'rows':n,'valid':valid,'failed_or_invalid':n-valid,
        'value_precision':tp/(tp+fp) if tp+fp else None,'value_recall':tp/(tp+fn) if tp+fn else None,
        'value_tp':tp,'value_fp':fp,'value_fn':fn,
        'structured_fields_correct':{k:sum(bool(r['metrics']['score'].get(k)) for r in rows) for k in STATE_FIELDS}}
    for k in ('complete_value_coverage','exact_values','strict_exact_state'):
        count=sum(r['metrics'][k] for r in rows);result[k]=count;result[k+'_rate_all']=count/n if n else None;result[k+'_rate_valid']=count/valid if valid else None
    return result

def supervised_tokens(tokenizer,row,max_length):
    """Exactly the inference prefix, then JSON+message-EOS; loss only on completion."""
    messages=row['messages'];assert [m['role'] for m in messages]==['system','user','assistant']
    prompt=tokenizer.apply_chat_template(messages[:2],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
    end=tokenizer.convert_tokens_to_ids('<|im_end|>');assert isinstance(end,int) and end!=tokenizer.unk_token_id
    completion=tokenizer.encode(messages[2]['content'],add_special_tokens=False)+[end]
    ids=prompt+completion
    if len(ids)>max_length:raise ValueError(f"{row['id']}: {len(ids)} tokens exceeds {max_length}; truncation is disabled")
    return ids,len(prompt)
