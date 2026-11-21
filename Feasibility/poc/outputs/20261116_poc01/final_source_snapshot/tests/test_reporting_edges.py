import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import *
from report import audit,paired_bootstrap
import pytest


def test_empty_and_invalid_candidates_are_explicit():
    empty=pd.DataFrame(columns=['run','risk','failure'])
    assert audit(empty)['status']=='empty_pool' and audit(empty)['AUROC'] is None
    bad=pd.DataFrame({'run':['a'],'risk':[np.nan],'failure':[False]})
    with pytest.raises(ValueError,match='Invalid'):audit(bad)
    bad=pd.DataFrame({'run':['a'],'risk':[.5],'failure':['incomplete']})
    with pytest.raises(ValueError,match='binary'):audit(bad)


def test_family_resampling_retains_paired_memberships():
    d=pd.DataFrame({'run':['a1','a2','b1','b2'],'family':[1,1,2,2],
                    'risk':[.9,.8,.2,.1],'failure':[1,1,0,0]})
    values,invalid=paired_bootstrap(d,d,[1,2])
    assert values and all(v==0 for v in values)
    assert 0<invalid<2000 and len(values)+invalid==2000


def test_official_validation_panels_have_all_predefined_groups():
    # The frozen outcome construction must have every class/background cell.
    from study import outcome_panel
    for difficulty in ['MAGNITUDE_LARGE','MAGNITUDE_MEDIUM']:
        # Validation is already unsealed development information, not held-out test.
        _,y,bg=outcome_panel(difficulty,'val')
        counts=pd.crosstab(y.numpy(),bg)
        assert counts.shape==(10,10) and counts.to_numpy().min()>0
