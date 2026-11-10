import copy,gzip,json
from pathlib import Path
import pytest
from adaptgym.extraction_eval import request_for,supervised_tokens,summarize

def row():
    return {'id':'case:1','messages':[{'role':'system','content':'extract'},
        {'role':'user','content':json.dumps({'visible_conversation':[{'id':'u1','content':'road'}],'previous_prediction':{'old':True}})},
        {'role':'assistant','content':'{"secret_current_target":true}'}]}

def test_scorer_target_never_enters_request():
    r=row();original=copy.deepcopy(r)
    prompt,payload=request_for(r)
    assert 'secret_current_target' not in prompt+json.dumps(payload)
    assert payload['previous_prediction']=={'old':True}
    assert request_for(r,{'predicted':True},True)[1]['previous_prediction']=={'predicted':True}
    assert request_for(r,None,True)[1]['previous_prediction'] is None
    assert r==original

def test_supervision_prefix_matches_inference_and_refuses_truncation():
    class Tokenizer:
        unk_token_id=None
        def apply_chat_template(self,messages,**kwargs):
            assert len(messages)==2 and kwargs=={'tokenize':True,'add_generation_prompt':True,'enable_thinking':False,'return_dict':False}
            return [1,2,3]
        def convert_tokens_to_ids(self,token):assert token=='<|im_end|>';return 9
        def encode(self,text,**kwargs):assert text==row()['messages'][-1]['content'];return [4,5]
    assert supervised_tokens(Tokenizer(),row(),6)==([1,2,3,4,5,9],3)
    with pytest.raises(ValueError,match='truncation is disabled'):supervised_tokens(Tokenizer(),row(),5)

def test_failures_remain_in_all_query_denominator():
    good={'score':{'valid':True},'value_tp':2,'value_fp':0,'value_fn':0,'complete_value_coverage':True,'exact_values':True,'strict_exact_state':True}
    bad={'score':{'valid':False},'value_tp':0,'value_fp':0,'value_fn':2,'complete_value_coverage':False,'exact_values':False,'strict_exact_state':False}
    result=summarize([{'metrics':good},{'metrics':bad}])
    assert result['complete_value_coverage_rate_all']==.5
    assert result['complete_value_coverage_rate_valid']==1
    assert result['value_recall']==.5 and result['failed_or_invalid']==1
