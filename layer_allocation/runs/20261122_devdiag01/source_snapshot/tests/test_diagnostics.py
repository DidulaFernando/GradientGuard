import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import diagnose as d
import numpy as np
import pandas as pd
import pytest

def test_family_variance_is_an_exact_partition_with_unequal_sizes():
    r=d.variance_parts([1,2,5,7,8],[0,0,1,1,1])
    assert r['total_ss']==pytest.approx(r['within_ss']+r['between_ss'])
    assert r['within_fraction']+r['between_fraction']==pytest.approx(1)

def test_family_constant_signal_has_no_within_variance():
    r=d.variance_parts([1,1,3,3],[0,0,1,1]);assert r['within_fraction']==0 and r['between_fraction']==1
    assert d.variance_parts([1,1],[0,1])['between_fraction'] is None

def test_within_concordance_orientation_and_ties():
    assert d.auc_within([1,0],[1,0])[0]==1
    assert d.auc_within([1,0],[0,1])[0]==0
    assert d.auc_within([1,0],[4,4])[0]==.5
    with pytest.raises(ValueError):d.auc_within([1,1],[0,1])
    with pytest.raises(ValueError):d.auc_within([2,0],[0,1])
    with pytest.raises(ValueError):d.auc_within([1,0],[float('nan'),1])

def test_family_offset_does_not_change_within_concordance():
    assert np.array_equal(d.auc_within([1,0,0],[3,1,2]),d.auc_within([1,0,0],[103,101,102]))

def test_fold_local_imputation_and_target_scaling():
    x=np.array([[1,np.nan],[3,4],[1000,9999],[2000,9999]],float);y=np.array([[2],[4],[100],[200]],float)
    p,b,u,a=d.fixed_reconstruction(x,y,[(np.array([0,1]),np.array([2,3])),(np.array([2,3]),np.array([0,1]))])
    assert a[0]['imputation_mean']==[2,4] and a[0]['target_mean']==[3] and a[0]['target_sd']==[1]
    assert b[2,0]==3 and u[2,0]==1 and np.isfinite(p).all()

def test_reconstruction_has_no_label_input():
    assert 'failure' not in d.fixed_reconstruction.__code__.co_varnames
    assert d.CFG['ridge_alpha']==1 and d.Ridge.__name__=='Ridge'

def test_multi_target_reconstruction_contract():
    x=np.arange(12,dtype=float).reshape(6,2);y=np.column_stack([x[:,0]+1,x[:,1]*2])
    folds=[(np.array([0,1,2]),np.array([3,4,5])),(np.array([3,4,5]),np.array([0,1,2]))]
    p,b,u,a=d.fixed_reconstruction(x,y,folds)
    assert p.shape==b.shape==u.shape==(6,2) and np.isfinite(p).all()

def test_flat_target_metrics_are_undefined_not_fake_perfect():
    r=d.reconstruction_scores(np.ones((4,1)),np.ones((4,1)),np.ones((4,1)),np.ones((4,1)))[0]
    assert r['R2'] is None and r['skill_vs_fold_training_mean'] is None and r['RMSE']==0

def test_family_folds_are_disjoint_and_cover_once():
    f=pd.DataFrame({'family':[0,0,1,1]})
    saved=[{'train_families':[0],'validation_families':[1]},{'train_families':[1],'validation_families':[0]}]
    folds=d.fold_indices(f,saved);assert len(folds)==2
    with pytest.raises(AssertionError):d.fold_indices(f,[{'train_families':[0,1],'validation_families':[1]}])

def test_conditional_permutation_preserves_counts_and_adjusts_maximum():
    meta,s=d.conditional_null([1,0,0,1,0],np.array([[5,5],[1,1],[0,0],[4,4],[2,2]]),[0,0,0,1,1])
    assert meta['assignments']==6 and meta['draws']==6 and meta['mode']=='exact'
    assert all(v['max24_adjusted_p']>=v['two_sided_conditional_p'] for v in s)
    assert s[0]['concordance']==1

def test_equal_family_weighting_not_pair_weighting():
    # First family has two winning pairs, second one losing pair: mean=0.5, not2/3.
    meta,s=d.conditional_null([1,0,0,1,0],np.array([[5],[1],[0],[1],[2]]),[0,0,0,1,1])
    assert s[0]['concordance']==.5

def test_no_mixed_families_explicitly_unavailable():
    assert d.conditional_null([1,1,0,0],np.ones((4,2)),[0,0,1,1])==(None,[])

def test_monte_carlo_is_reproducible_and_counted(monkeypatch):
    monkeypatch.setitem(d.CFG,'permutation_exact_limit',1)
    monkeypatch.setitem(d.CFG,'permutation_draws',19)
    y=[1,0,0,1,0];x=np.arange(10).reshape(5,2);g=[0,0,0,1,1]
    a=d.conditional_null(y,x,g);b=d.conditional_null(y,x,g)
    assert a==b and a[0]['draws']==19 and a[0]['mode']=='monte_carlo'

def test_semantic_allowlist_blocks_evaluation_and_test_outcomes():
    for p in [d.SRC/'evaluation_features.csv',d.SRC/'evaluation_predictions.csv',d.SRC/'terminal_outcomes.csv',d.SRC/'outcomes/test/fresh_f02_c0_s0.json']:
        with pytest.raises(ValueError):d.check_input(p)

def test_primary_schema_and_no_risk_training_api():
    assert d.Z==[f'{b}_share_{s}' for b in ['shallow','deep'] for s in ['last','mean','std']]
    assert not any(n in d.__dict__ for n in ['LogisticRegression','HistGradientBoostingClassifier','torch','joblib'])

def test_protection_hashes_and_nonfinite_json(tmp_path):
    p=tmp_path/'fixture.json';d.write(p,{'a':np.float64(np.nan),'b':np.int64(3)});assert d.read(p)=={'a':None,'b':3}
    h=d.sha(p);p.write_text('changed');assert h!=d.sha(p)

def test_paired_panels_source_id_counts_without_outcome_read():
    # Source-schema check only; no diagnostic statistical results computed here.
    f=pd.read_csv(d.SRC/'development_features.csv',usecols=['run','panel','prefix','split'])
    q=f[f.prefix==.1];assert set(q.split)=={'development'}
    assert len(set(q.run))==124 and set(q[q.panel=='A'].run)==set(q[q.panel=='B'].run)
