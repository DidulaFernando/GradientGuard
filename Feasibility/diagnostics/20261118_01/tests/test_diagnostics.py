"""Synthetic audit fixtures and read-only checks of already fitted scientific models."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from core import *
import pytest,joblib
from slopes import layout,replace_slopes


def toy(labels):
    return pd.DataFrame({'run':[f'r{i:02d}' for i in range(len(labels))],'risk':np.linspace(.9,.1,len(labels)),'failure':labels})


@pytest.mark.parametrize('labels',[[1,0,1,0],[0,0,0,0],[1,1,1,1],[]])
def test_curve_bounds_and_edge_cases(labels):
    d=toy(labels);c=curve(d);f=sum(labels)
    assert c.k.tolist()==list(range(len(labels)+1))
    assert c.discoveries.is_monotonic_increasing
    assert c.discoveries.iloc[-1]==f
    assert (c.discoveries<=np.minimum(c.k,f)).all() and (c.discoveries>=0).all()
    assert np.isnan(c.precision.iloc[0])
    if not f:assert c.recall.isna().all()
    if len(labels):assert np.allclose(c.random_expected,c.k*f/len(labels))


def test_ties_and_same_candidates():
    a=pd.DataFrame({'run':['b','a','d','c'],'risk':[.5]*4,'failure':[0,1,1,0]})
    assert ranked(a).run.tolist()==['a','b','c','d']
    assert curve(a).iloc[1].discoveries==1
    assert compare_sets(a,a.iloc[::-1],2)['selected_set_identical']
    with pytest.raises(AssertionError):compare_sets(a,a.iloc[:2],1)


@pytest.mark.parametrize('bad',['nan_risk','missing_label','nonbinary','duplicate','missing_id'])
def test_invalid_inputs_fail_closed(bad):
    a=toy([0,1,0,1])
    if bad=='nan_risk':a.loc[0,'risk']=np.nan
    if bad=='missing_label':a.loc[0,'failure']=np.nan
    if bad=='nonbinary':a['failure']=a.failure.astype(object);a.loc[0,'failure']='incomplete'
    if bad=='duplicate':a.loc[0,'run']=a.loc[1,'run']
    if bad=='missing_id':a.loc[0,'run']=None
    with pytest.raises(ValueError):curve(a)


def test_fractional_budget_after_family_removal():
    assert budget(36,'1/4')==9 and budget(36,'1/3')==12
    assert budget(32,'1/4')==8 and budget(32,'1/3')==10
    assert budget(20,'7/20')==7


def test_family_draw_keeps_nested_runs_together():
    fam=np.array([1,1,1,1,2,2,2,2]);ids=np.array([f'r{i}' for i in range(8)])
    ix,copies=resample_indices(fam,ids,[1,1,2,0])
    assert ix.tolist()==list(range(4))+list(range(4))+list(range(4,8))
    assert len(set(copies))==12 and all(str(c).endswith('_2') for c in copies[-4:])


def test_exact_original_and_panel_cohorts():
    groups,evaluation,families,original=load_inputs()
    assert len(families)==10 and len(evaluation)==40
    assert all(r['discoveries']==8 for r in original)
    for (cohort,v,m),d in groups.items():
        assert set(d.run)==set(groups[(cohort,'A','E')].run)
        if cohort=='eligible':assert len(d)==36 and d.failure.sum()==12


def test_no_fitting_guard():
    from sklearn.linear_model import LogisticRegression
    with no_fitting():
        with pytest.raises(RuntimeError,match='prohibited'):LogisticRegression().fit([[0],[1]],[0,1])


def test_actual_slope_lineage_and_reference_replacement():
    models=joblib.load(POC/'pipelines.joblib');before=joblib.hash(models)
    model,cols=models['eligible_0.1']['logistic_L8']
    data=features('evaluation','A');data=data[data.eligible];x=data[cols]
    names,_,support,final=layout(model,cols)
    assert all(s in names for s in SLOPES)
    altered,refs=replace_slopes(model,cols,data,SLOPES)
    assert altered[[n for n in cols if n not in SLOPES]].equals(x[[n for n in cols if n not in SLOPES]])
    z=model[:-1].transform(x);zz=model[:-1].transform(altered)
    delta=np.zeros(len(x))
    for name in SLOPES:
        assert refs[name]==model.named_steps['standardscaler'].mean_[names.index(name)]
        if name in final:
            j=final.index(name);assert np.max(abs(zz[:,j]))<1e-12
            delta-=model.steps[-1][1].coef_[0,j]*z[:,j]
    assert np.allclose(model.decision_function(altered)-model.decision_function(x),delta,atol=1e-10,rtol=1e-10)
    assert before==joblib.hash(models)


def test_preservation_and_frozen_inputs():
    guard();assert verify_prior('tests')['passed']
