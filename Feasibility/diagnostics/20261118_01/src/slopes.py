from core import *
import joblib
from scipy.special import expit
from scipy.stats import spearmanr,kendalltau
from sklearn.metrics import average_precision_score,roc_auc_score


def layout(model,columns):
    imputer=model.named_steps['simpleimputer'];scaler=model.named_steps['standardscaler'];selector=model.named_steps['variancethreshold']
    names=list(imputer.get_feature_names_out(columns));scaled_names=list(scaler.get_feature_names_out(names))
    support=selector.get_support();final=list(selector.get_feature_names_out(scaled_names))
    assert list(columns)==list(imputer.feature_names_in_)
    assert len(support)==len(names)==len(scaler.mean_)
    return names,scaled_names,support,final


def replace_slopes(model,columns,frame,targets):
    names,_,_,_=layout(model,columns);out=frame[columns].copy();refs={}
    for name in targets:
        if name not in names:raise ValueError('Training reference unavailable after imputation')
        j=names.index(name);ref=float(model.named_steps['standardscaler'].mean_[j])
        if not np.isfinite(ref):raise ValueError('Nonfinite training reference')
        # The original fitted imputer has no feature expansion and returns this numeric column unchanged when finite.
        assert not model.named_steps['simpleimputer'].add_indicator
        out[name]=ref;refs[name]=ref
    return out,refs


def parameter_rows(model,columns):
    names,_,support,final=layout(model,columns);est=model.steps[-1][1]
    coefficient=dict(zip(final,est.coef_[0]))
    return {name:{'coefficient':float(coefficient.get(name,0.)),'retained':name in final,
                  'training_mean':float(model.named_steps['standardscaler'].mean_[names.index(name)]),
                  'training_scale':float(model.named_steps['standardscaler'].scale_[names.index(name)])}
            for name in columns if name in names}


def run_slopes(groups):
    cfg=read(CONFIG);tol=cfg['parity'];frozen=read(POC/'predictor_freeze.json')
    assert sha(POC/'pipelines.joblib')==frozen['pipelines_sha256']
    models=joblib.load(POC/'pipelines.joblib');before=joblib.hash(models)
    frames={(split,panel):features(split,panel) for split in ['development','evaluation'] for panel in ['A','B']}
    feature_parity=[]
    for split in ['development','evaluation']:
        old=pd.read_csv(POC/f'{split}_features_0.1.csv').set_index('run')
        new=frames[(split,'A')].set_index('run').loc[old.index]
        assert list(new.columns)==list(old.columns)
        for col in old:
            if pd.api.types.is_numeric_dtype(old[col]):
                a=new[col].to_numpy();b=old[col].to_numpy()
                assert np.allclose(a,b,atol=tol['atol'],rtol=tol['rtol'],equal_nan=True)
                err=float(np.nanmax(np.abs(a.astype(float)-b.astype(float))))
            else:assert new[col].equals(old[col]);err=0.
            feature_parity.append({'split':split,'column':col,'max_absolute_error':err})
    lineage=[];variation=[];coefficients=[];contributions=[];prediction_parity=[];comparisons=[];common=[]
    replacements=[];replacement_effects=[];replacement_curves=[];native_preds={}
    with no_fitting() as calls:
        for cohort in cfg['cohorts']:
            for variant in cfg['variants']:
                v=variant['name'];collection=cohort+'_0.1'+('_B' if variant['refit_B'] else '')
                audit_models=[k[2] for k in groups if k[:2]==(cohort,v)]
                for label in audit_models:
                    model,columns=models[collection][label]
                    data=groups[(cohort,v,label)];frame=frames[('evaluation',variant['panel'])].set_index('run').loc[data.run]
                    assert np.array_equal(frame.eligible.to_numpy(),data.eligible.to_numpy())
                    risk=model.predict_proba(frame[columns])[:,1]
                    err=float(np.max(abs(risk-data.risk.to_numpy())))
                    assert np.allclose(risk,data.risk,atol=tol['atol'],rtol=tol['rtol']),f'Frozen probability mismatch: {cohort}/{v}/{label}: {err}'
                    native_preds[(cohort,v,label)]=risk
                    prediction_parity.append({'cohort':cohort,'variant':v,'model':label,'N':len(data),'max_absolute_error':err,'passed':True})
                for label in ['logistic_L6','logistic_L8']:
                    model,columns=models[collection][label];data=groups[(cohort,v,label)]
                    frame=frames[('evaluation',variant['panel'])].set_index('run').loc[data.run]
                    names,scaled_names,support,final=layout(model,columns)
                    transformed=model[:-1].transform(frame[columns]);params=parameter_rows(model,columns)
                    for j,name in enumerate(columns):
                        lineage.append({'cohort':cohort,'variant':v,'model':label,'input_column':name,'input_index':j,
                            'imputed_index':names.index(name) if name in names else None,
                            'scaled_index':scaled_names.index(name) if name in scaled_names else None,
                            'retained':name in final,'estimator_index':final.index(name) if name in final else None,
                            'coefficient':params.get(name,{}).get('coefficient'),
                            'C':float(model.steps[-1][1].C)})
                    if label=='logistic_L8':
                        per_run=np.zeros((len(frame),2));training=frames[('development',variant['panel'])]
                        if cohort=='eligible':training=training[training.eligible]
                        for si,name in enumerate(SLOPES):
                            p=params[name];retained=name in final
                            z=transformed[:,final.index(name)] if retained else np.zeros(len(frame))
                            component=p['coefficient']*z;per_run[:,si]=component
                            raw=frame[name].to_numpy(float)
                            for split,values in [('development',training[name].to_numpy(float)),('evaluation',raw)]:
                                finite=values[np.isfinite(values)]
                                variation.append({'cohort':cohort,'variant':v,'feature':name,'split':split,'N':len(values),
                                    'missing':int(np.isnan(values).sum()),'nonfinite':int((~np.isfinite(values)).sum()),
                                    'distinct_finite':len(np.unique(finite)),'minimum':float(finite.min()) if len(finite) else np.nan,
                                    'maximum':float(finite.max()) if len(finite) else np.nan,'mean':float(finite.mean()) if len(finite) else np.nan,
                                    'population_sd':float(finite.std()) if len(finite) else np.nan})
                            coefficients.append({'cohort':cohort,'variant':v,'feature':name,**p,
                                'imputation_statistic':float(model.named_steps['simpleimputer'].statistics_[columns.index(name)]),
                                'fitted_variance':float(model.named_steps['standardscaler'].var_[names.index(name)]),
                                'variance_filter_value':float(model.named_steps['variancethreshold'].variances_[names.index(name)]),
                                'standardized_min':float(z.min()),'standardized_max':float(z.max()),
                                'contribution_mean':float(component.mean()),'contribution_mean_absolute':float(np.abs(component).mean()),
                                'contribution_max_absolute':float(np.abs(component).max())})
                        logit=model.decision_function(frame[columns]);assert np.allclose(expit(logit),native_preds[(cohort,v,label)],atol=tol['atol'],rtol=tol['rtol'])
                        for i,r in enumerate(data.itertuples()):
                            contributions.append({'cohort':cohort,'variant':v,'run':r.run,'risk':r.risk,'logit':float(logit[i]),
                                'shallow_slope_component':float(per_run[i,0]),'deep_slope_component':float(per_run[i,1]),
                                'combined_slope_component':float(per_run[i].sum())})
                        for mode,targets in [('shallow_share_slope',[SLOPES[0]]),('deep_share_slope',[SLOPES[1]]),('both',SLOPES)]:
                            altered,refs=replace_slopes(model,columns,frame,targets)
                            assert altered[[c for c in columns if c not in targets]].equals(frame[[c for c in columns if c not in targets]])
                            newrisk=model.predict_proba(altered)[:,1];newlogit=model.decision_function(altered)
                            expected=-per_run[:,[SLOPES.index(n) for n in targets]].sum(1)
                            assert np.allclose(newlogit-logit,expected,atol=tol['logit_atol'],rtol=tol['rtol'])
                            shifted=model[:-1].transform(altered)
                            for name in targets:
                                if name in final:assert np.max(abs(shifted[:,final.index(name)]))<=tol['atol']
                            modified=data.copy();modified['risk']=newrisk
                            c=curve(modified).assign(cohort=cohort,variant=v,replacement=mode)
                            replacement_curves.append(c)
                            for i,r in enumerate(data.itertuples()):
                                replacements.append({'cohort':cohort,'variant':v,'replacement':mode,'run':r.run,'family':r.family,
                                    'failure':r.failure,'original_risk':r.risk,'replacement_risk':float(newrisk[i]),
                                    'reference_shallow':refs.get(SLOPES[0]),'reference_deep':refs.get(SLOPES[1])})
                            for k in [9,12,budget(len(data),'1/4')]:
                                identity=compare_sets(data,modified,k)
                                replacement_effects.append({'cohort':cohort,'variant':v,'replacement':mode,'k':k,
                                    'max_absolute_probability_change':float(np.max(abs(newrisk-native_preds[(cohort,v,label)]))),
                                    'mean_absolute_probability_change':float(np.mean(abs(newrisk-native_preds[(cohort,v,label)]))),
                                    'max_logit_identity_error':float(np.max(abs(newlogit-logit-expected))),
                                    'original_discoveries':int(curve(data).iloc[k].discoveries),
                                    'replacement_discoveries':int(c.iloc[k].discoveries),**identity})
                m6,c6=models[collection]['logistic_L6'];m8,c8=models[collection]['logistic_L8']
                p6=parameter_rows(m6,c6);p8=parameter_rows(m8,c8)
                for name in c6:
                    common.append({'cohort':cohort,'variant':v,'feature':name,'coefficient_L6':p6[name]['coefficient'],
                        'coefficient_L8':p8[name]['coefficient'],'coefficient_difference':p8[name]['coefficient']-p6[name]['coefficient'],
                        'training_mean_difference':p8[name]['training_mean']-p6[name]['training_mean'],
                        'training_scale_difference':p8[name]['training_scale']-p6[name]['training_scale'],
                        'C_L6':float(m6.steps[-1][1].C),'C_L8':float(m8.steps[-1][1].C)})
                a=groups[(cohort,v,'logistic_L6')];b=groups[(cohort,v,'logistic_L8')]
                ra=a.risk.to_numpy();rb=b.risk.to_numpy()
                for k in range(len(a)+1):
                    comparisons.append({'cohort':cohort,'variant':v,'k':k,
                        'probabilities_exact':np.array_equal(ra,rb),'probabilities_close':np.allclose(ra,rb,atol=tol['atol'],rtol=tol['rtol']),
                        'max_absolute_probability_difference':float(np.max(abs(ra-rb))),
                        'mean_absolute_probability_difference':float(np.mean(abs(ra-rb))),
                        'spearman_risk':float(spearmanr(ra,rb).statistic),'kendall_tau_risk':float(kendalltau(ra,rb).statistic),
                        'AP_L6':float(average_precision_score(a.failure,ra)),'AP_L8':float(average_precision_score(b.failure,rb)),
                        'AUROC_L6':float(roc_auc_score(a.failure,ra)),'AUROC_L8':float(roc_auc_score(b.failure,rb)),
                        **compare_sets(a,b,k)})
        assert not calls
    assert before==joblib.hash(models),'Inference modified fitted pipelines'
    assert sha(POC/'pipelines.joblib')==frozen['pipelines_sha256']
    for name,records in [('feature_parity',feature_parity),('prediction_parity',prediction_parity),('preprocessing_lineage',lineage),
        ('feature_variation',variation),('slope_coefficients',coefficients),('per_run_slope_contributions',contributions),
        ('common_coefficients',common),('L6_L8_comparisons',comparisons),('reference_predictions',replacements),('reference_effects',replacement_effects)]:save_table(name,records)
    save_table('reference_curves',pd.concat(replacement_curves,ignore_index=True))
    write(ATTEMPT/'reports/pipeline_immutability.json',{'passed':True,'in_memory_before':before,'in_memory_after':joblib.hash(models),
          'fit_calls':0,'predictors_retrained':0,'pipeline_file_sha256':sha(POC/'pipelines.joblib')})
    event('slopes_complete',prediction_parity_checks=len(prediction_parity),lineage_rows=len(lineage))
