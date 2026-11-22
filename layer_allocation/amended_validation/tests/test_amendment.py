from test_fresh import BASE,HOME,read,pd,cfg

def test_original_support_and_unavailable_identities():
    previous=BASE/'runs/20261121_fresh01'
    ready=read(previous/'readiness.json')
    assert ready['N']==107 and ready['families']==29
    assert ready['outcomes']=={'0':{'models':69,'families':19},'1':{'models':38,'families':14}}
    cohort=pd.read_csv(previous/'development_readiness_cohort.csv')
    assert int((cohort.groupby('family').failure.nunique()>1).sum())==4
    affected=[r for p in (previous/'preparation_failures').glob('*.json') for r in read(p)['affected_runs']]
    assert len(affected)==len(set(affected))==4
    assert len(list((previous/'classifiers').glob('*/complete.json')))==124

def test_no_evaluation_outcomes_or_assertion_override():
    assert not (BASE/'runs/20261121_fresh01/outcomes/test').exists()
    source=(HOME/'ggfresh/measure_data.py').read_text()
    assert 'skip_group_validation' not in source
    training=(HOME/'ggfresh/training.py').read_text()
    assert "if not str(error).startswith('No examples in '):raise" in training
    assert 'preflight_evaluation' in training

def test_only_execution_requirement_amended():
    original=read(BASE/'fresh_validation/configs/fresh.json');amended=cfg()
    for k in original:
        if k not in ['attempt','readiness_rule']:assert original[k]==amended[k],k
