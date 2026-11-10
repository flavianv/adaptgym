#!/usr/bin/env python3
"""Rescore saved outputs only; no generation, extraction or API calls."""
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from adaptgym.feature_scoring import POLICY,score_features,summarize_features

def rescore_directory(directory):
    directory=Path(directory)
    index=json.loads((directory/'index.json').read_text());all_scores={}
    for entry in index['cases']:
        file=directory/entry['file'];bundle=json.loads(file.read_text())
        for model,modes in bundle['models'].items():
            for mode,run in modes.items():
                scores=[]
                for cp in run['checkpoints']:
                    score=score_features(cp.get('profile'),cp['reference_profile']);cp['feature_score']=score;scores.append(score)
                    cp.setdefault('original_metrics',copy.deepcopy(cp['metrics']))
                    cp['metrics'].update(value_precision=score['precision'],value_recall=score['recall'],feature_pass=score['accepted'])
                run['feature_summary']=summarize_features(scores);all_scores.setdefault(model,{}).setdefault(mode,[]).extend(scores)
        file.write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n')
    index['feature_policy']={'id':POLICY,'matching':'Exact feature, operator and scalar values; strings lowercase only; array values are unordered sets.',
        'errors':'Missing or changed target features/values. No aliases or cross-feature equivalence.',
        'warnings':'All extra features/values; warnings never block acceptance.',
        'diagnostics':'Scope, strength, evidence, item/task state and native validity are separate diagnostics, not feature-value gates.'}
    index['feature_summary']={model:{mode:summarize_features(scores) for mode,scores in modes.items()} for model,modes in all_scores.items()}
    (directory/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n')
    return index['feature_summary']

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=ROOT/'examples/model_comparison');a=p.parse_args()
    print(json.dumps(rescore_directory(a.directory),indent=2))
if __name__=='__main__':main()
