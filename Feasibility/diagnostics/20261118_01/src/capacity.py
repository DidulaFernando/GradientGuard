from core import *


def run_curves(groups,evaluation,families):
    rows=[];selections=[];contrasts=[];marked=[]
    for (cohort,variant,model),d in groups.items():
        c=curve(d);meta={'cohort':cohort,'variant':variant,'model':model}
        rows.append(c.assign(**meta))
        order=ranked(d)
        for rank,r in enumerate(order.itertuples(),1):
            selections.append({**meta,'rank':rank,'run':r.run,'family':r.family,'risk':r.risk,'failure':bool(r.failure),
              'selected_original_25':rank<=budget(len(d),'1/4'),'selected_nine':rank<=9,'selected_twelve':rank<=12})
        for label,k in [('original_25',budget(len(d),'1/4')),('nine',9),('twelve',12)]:
            marked.append({**meta,'mark':label,**c.iloc[k].to_dict()})
    all_curves=pd.concat(rows,ignore_index=True)
    for (cohort,variant),g in all_curves.groupby(['cohort','variant']):
        primary=g[g.model=='logistic_L8'].set_index('k')
        for control,b in g.groupby('model'):
            if control=='logistic_L8':continue
            b=b.set_index('k');diff=primary.discoveries-b.discoveries
            for k,val in diff.items():
                contrasts.append({'cohort':cohort,'variant':variant,'control':control,'N':int(primary.loc[k,'N']),
                    'k':k,'fraction':primary.loc[k,'fraction'],'L8_discoveries':int(primary.loc[k,'discoveries']),
                    'control_discoveries':int(b.loc[k,'discoveries']),'difference':int(val),
                    'Delta20':20*val/primary.loc[k,'N']})
    save_table('all_budget_curves',all_curves);save_table('audit_selections',selections)
    save_table('all_budget_contrasts',contrasts);save_table('marked_budgets',marked)
    headroom=[]
    for (cohort,variant,model),d in groups.items():
        if model!='E':continue
        c=curve(d)
        for k in [9,12,budget(len(d),'1/4')]:
            r=c.iloc[k];headroom.append({'cohort':cohort,'variant':variant,'k':k,
                'E_discoveries':int(r.discoveries),'oracle_ceiling':int(r.oracle_ceiling),
                'remaining':int(r.headroom),'maximum_Delta20':20*r.headroom/len(d),
                'reference_target_1_attainable':bool(20*r.headroom/len(d)>=1),
                'target_is_original_only':k==budget(len(d),'1/4')})
    save_table('headroom',headroom)
    div=[]
    for cohort in ['eligible','full']:
        z=evaluation[evaluation.eligible] if cohort=='eligible' else evaluation
        for f in families:
            d=z[z.family==f]
            div.append({'cohort':cohort,'family':f,'N':len(d),'failures':int(d.failure.sum()),
                        'nonfailures':int((~d.failure).sum()),'mixed_outcomes':d.failure.nunique()==2,
                        'constructions':d.construction.nunique()})
    save_table('family_diversity',div)
    event('curves_complete',rows=len(all_curves))


def run_stability(groups,families):
    cfg=read(CONFIG);fractions=cfg['fractions'];lofo=[];intervals=[];parity=[]
    for cohort in cfg['cohorts']:
        variants=[v['name'] for v in cfg['variants']]
        models=[k[2] for k in groups if k[0]==cohort and k[1]=='A']
        for variant in variants:
            a=groups[(cohort,variant,'logistic_L8')]
            for control in [m for m in models if m!='logistic_L8']:
                b=groups[(cohort,variant,control)]
                for f in families:
                    aa=a[a.family!=f];bb=b[b.family!=f];ca=curve(aa);cb=curve(bb)
                    views=[('fraction','1/4',budget(len(aa),'1/4')),('fraction','1/3',budget(len(aa),'1/3')),
                           ('fixed_count','9',min(9,len(aa))),('fixed_count','12',min(12,len(aa)))]
                    for kind,requested,k in views:
                        da=int(ca.iloc[k].discoveries);db=int(cb.iloc[k].discoveries)
                        lofo.append({'cohort':cohort,'variant':variant,'control':control,'omitted_family':f,
                            'removed_candidates':len(a)-len(aa),'view':kind,'requested':requested,'N':len(aa),'k':k,
                            'actual_fraction':k/len(aa) if len(aa) else np.nan,'L8_discoveries':da,'control_discoveries':db,
                            'difference':da-db,'Delta20':20*(da-db)/len(aa) if len(aa) else np.nan})
        # Match the original seed, family list, duplicate identifier rule and invalid rules.
        basis=groups[(cohort,'A','E')].sort_values('run')
        ids=basis.run.to_numpy();fam=basis.family.to_numpy();labels=basis.failure.to_numpy(int)
        by={f:np.flatnonzero(fam==f) for f in families}
        risks={(v,m):groups[(cohort,v,m)].set_index('run').loc[ids,'risk'].to_numpy() for v in variants for m in models}
        results={(v,m):[] for v in variants for m in models if m!='logistic_L8'}
        raw={key:[] for key in results};invalid=0;draws=[]
        rng=np.random.default_rng(cfg['bootstrap']['seed'])
        for replicate in range(cfg['bootstrap']['replicates']):
            draw=rng.choice(families,len(families),replace=True)
            if cohort=='eligible':draws.extend({'replicate':replicate,'draw_position':j,'family':int(f)} for j,f in enumerate(draw))
            ix,copy_ids=resample_indices(fam,ids,draw)
            if len(ix)==0 or len(set(labels[ix]))<2:invalid+=1;continue
            ks=np.array([budget(len(ix),fraction) for fraction in fractions])
            discovered={}
            for key,risk in risks.items():
                order=np.lexsort((copy_ids,-risk[ix]));discovered[key]=np.r_[0,np.cumsum(labels[ix][order])][ks]
            for key in results:
                v,m=key;diff=discovered[(v,'logistic_L8')]-discovered[key]
                raw[key].append(diff);results[key].append(20*diff/len(ix))
        if cohort=='eligible':save_table('bootstrap_family_draws',draws)
        for (v,m),values in results.items():
            arr=np.asarray(values);rawarr=np.asarray(raw[(v,m)])
            full=groups[(cohort,v,'logistic_L8')];ca=curve(full);cb=curve(groups[(cohort,v,m)])
            for j,fraction in enumerate(fractions):
                k=budget(len(full),fraction);gain=int(ca.iloc[k].discoveries)-int(cb.iloc[k].discoveries)
                intervals.append({'cohort':cohort,'variant':v,'control':m,'fraction_rule':fraction,
                    'N':len(full),'k':k,'actual_fraction':k/len(full),'difference':gain,'Delta20':20*gain/len(full),
                    'interval_low':np.quantile(arr[:,j],.025) if len(arr) else np.nan,
                    'interval_high':np.quantile(arr[:,j],.975) if len(arr) else np.nan,
                    'difference_low':np.quantile(rawarr[:,j],.025) if len(arr) else np.nan,
                    'difference_high':np.quantile(rawarr[:,j],.975) if len(arr) else np.nan,
                    'valid_resamples':len(values),'invalid_resamples':invalid,'sampled_families':len(families)})
    intervals=pd.DataFrame(intervals);lofo=pd.DataFrame(lofo)
    old=pd.read_csv(POC/'reports/contrasts.csv')
    oldlofo=pd.read_csv(POC/'reports/leave_one_family_out.csv')
    variants={v['name']:v for v in cfg['variants']}
    for r in intervals[intervals.fraction_rule=='1/4'].itertuples():
        v=variants[r.variant];d=old[(old.cohort==r.cohort)&(old.prefix==.1)&(old.panel==v['panel'])&(old.refit_B==v['refit_B'])&(old.control==r.control)]
        if d.empty:continue
        expected=d.iloc[0];err=max(abs(r.interval_low-expected.interval_low),abs(r.interval_high-expected.interval_high))
        assert err<=cfg['parity']['atol'] and r.invalid_resamples==expected.invalid_resamples
        parity.append({'cohort':r.cohort,'variant':r.variant,'control':r.control,'max_interval_error':err,'invalid_resamples_match':True})
    for r in lofo[(lofo.view=='fraction')&(lofo.requested=='1/4')].itertuples():
        v=variants[r.variant];d=oldlofo[(oldlofo.cohort==r.cohort)&(oldlofo.prefix==.1)&(oldlofo.panel==v['panel'])&(oldlofo.refit_B==v['refit_B'])&(oldlofo.control==r.control)&(oldlofo.omitted_family==r.omitted_family)]
        if len(d):assert abs(r.Delta20-d.iloc[0].Delta20)<=cfg['parity']['atol']
    save_table('bootstrap_intervals',intervals);save_table('leave_one_family_out',lofo);save_table('original_uncertainty_parity',parity)
    event('stability_complete',intervals=len(intervals),lofo=len(lofo),original_intervals_reproduced=len(parity))
