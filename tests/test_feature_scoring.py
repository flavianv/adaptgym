from adaptgym.feature_scoring import score_features,summarize_features

def profile(*claims,**other):return {'claims':list(claims),**other}
def claim(attribute,value,operator='eq',**metadata):return dict(attribute=attribute,value=value,operator=operator,**metadata)

def test_lowercase_exact_without_synonym_or_whitespace_forgiveness():
    target=profile(claim('brand','Adidas'))
    assert score_features(profile(claim('BRAND','ADIDAS')),target)['accepted']
    assert not score_features(profile(claim('brand','Adidas ')),target)['accepted']
    assert not score_features(profile(claim('brand','Adidas Originals')),target)['accepted']

def test_extra_only_warns_and_does_not_block():
    result=score_features(profile(claim('surface','ROAD'),claim('sport','running')),profile(claim('surface','road')))
    assert result['accepted'] and result['error_count']==0 and result['warning_count']==1
    assert result['precision']==.5 and result['recall']==1

def test_cross_feature_encoding_is_missing_plus_extra():
    result=score_features(profile(claim('sport','basketball')),profile(claim('primary_use','Basketball')))
    assert not result['accepted'] and result['errors'][0]['type']=='missing'
    assert result['errors'][0]['attribute']=='primary_use' and result['warnings'][0]['attribute']=='sport'

def test_range_operator_and_numeric_type_are_preserved():
    target=profile(claim('price',100,'lte'))
    assert score_features(profile(claim('price',100.0,'lte')),target)['accepted']
    assert not score_features(profile(claim('price',100,'eq')),target)['accepted']
    assert not score_features(profile(claim('price','100','lte')),target)['accepted']
    assert not score_features(profile(claim('price',True,'lte')),profile(claim('price',1,'lte')))['accepted']

def test_array_values_are_sets_with_missing_errors_and_extra_warnings():
    target=profile(claim('colour',['Black','Red'],'in'))
    assert score_features(profile(claim('colour',['RED','black','red'],'in')),target)['accepted']
    partial=score_features(profile(claim('colour',['black','blue'],'in')),target)
    assert not partial['accepted'] and partial['tp']==1 and partial['fp']==1 and partial['fn']==1

def test_unrelated_invalid_relations_do_not_zero_correct_features():
    target=profile(claim('brand','Adidas'),claim('size',42))
    actual=profile(claim('brand','ADIDAS'),claim('size',42),preferences=[{'preferred':'unseen','over':'also-unseen'}])
    result=score_features(actual,target)
    assert result['accepted'] and result['tp']==2 and result['recall']==1
    assert not score_features(None,target)['accepted']
    assert summarize_features([result,score_features(None,target)])['recall']==.5


def test_replacement_is_one_changed_error_not_a_duplicate_extra_warning():
    result=score_features(profile(claim('brand','Nike')),profile(claim('brand','Adidas')))
    assert result['error_count']==1 and result['errors'][0]['type']=='changed'
    assert result['warning_count']==0 and result['fp']==1 and result['fn']==1
