"""Exact target-feature scoring: missing/changed are errors; extras are warnings."""
import json
from decimal import Decimal

POLICY='exact-features-lowercase-v1'

def atom(value):
    if isinstance(value,str):return ('string',value.lower())
    if isinstance(value,bool):return ('boolean',value)
    if isinstance(value,(int,float)):return ('number',Decimal(str(value)).normalize())
    if value is None:return ('null',None)
    return ('json',json.dumps(value,sort_keys=True,ensure_ascii=False))

def features(profile):
    result={}
    claims=(profile or {}).get('claims',[])
    if not isinstance(claims,list):return result
    for claim in claims:
        if not isinstance(claim,dict) or not isinstance(claim.get('attribute'),str):continue
        attribute=claim['attribute'].lower();operator=claim['operator'].lower() if isinstance(claim.get('operator'),str) else '<missing>'
        values=claim.get('value');values=values if isinstance(values,list) else [values]
        for value in values:
            key=(attribute,operator,atom(value))
            result[key]={'attribute':attribute,'operator':operator,'value':value.lower() if isinstance(value,str) else value}
    return result

def score_features(prediction,reference):
    expected=features(reference);actual=features(prediction if isinstance(prediction,dict) else None)
    missing=expected.keys()-actual.keys();extra=actual.keys()-expected.keys();errors=[];warnings=[]
    for attribute in sorted({k[0] for k in missing}):
        got=[v for k,v in actual.items() if k[0]==attribute]
        errors.append({'type':'changed' if got else 'missing','attribute':attribute,
                       'expected':[v for k,v in expected.items() if k[0]==attribute],
                       'extracted':got,'missing':[expected[k] for k in sorted(missing,key=str) if k[0]==attribute]})
    changed_attributes={e['attribute'] for e in errors}
    for attribute in sorted({k[0] for k in extra}-changed_attributes):
        warnings.append({'type':'invented','attribute':attribute,'values':[actual[k] for k in sorted(extra,key=str) if k[0]==attribute]})
    usable=isinstance(prediction,dict) and isinstance(prediction.get('claims'),list)
    if not usable and not errors:errors.append({'type':'missing','attribute':'output','expected':'structured extraction','extracted':None})
    tp=len(expected.keys()&actual.keys());fp=len(extra);fn=len(missing)
    return {'policy':POLICY,'accepted':usable and not errors,'output_present':usable,
            'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,'errors':errors,'warnings':warnings,
            'error_count':len(errors),'warning_count':len(warnings)}

def summarize_features(scores):
    scores=list(scores);tp=sum(s['tp'] for s in scores);fp=sum(s['fp'] for s in scores);fn=sum(s['fn'] for s in scores)
    return {'policy':POLICY,'turns':len(scores),'passed_turns':sum(s['accepted'] for s in scores),
            'error_turns':sum(not s['accepted'] for s in scores),'warning_turns':sum(bool(s['warnings']) for s in scores),
            'errors':sum(s['error_count'] for s in scores),'warnings':sum(s['warning_count'] for s in scores),
            'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None}
