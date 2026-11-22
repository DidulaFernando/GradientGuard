"""Plot complete predefined curves from saved scalar outputs; no model fitting."""
from pathlib import Path
import csv, os

BASE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(BASE/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    o=BASE/'runs/20261122_fresh02';r=BASE/'reports/20261122_fresh02'
    with (o/'discovery_curves.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    def curve(cohort,variant,model):
        data=[p for p in rows if p['cohort']==cohort and p['prefix']=='0.1' and p['variant']==variant and p['model']==model]
        data.sort(key=lambda p:int(p['k']))
        return [int(p['k']) for p in data],[int(p['discoveries']) for p in data]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    banks=[['E','logistic_B','logistic_L2','logistic_L6'],['logistic_N','logistic_NZ','logistic_M','logistic_MZ']]
    for col,cohort in enumerate(['eligible','full']):
        for row,methods in enumerate(banks):
            ax=axes[row,col]
            for m in methods:
                x,y=curve(cohort,'A',m)
                ax.step(x,y,where='post',label=m.replace('logistic_',''),linestyle='--' if m.endswith('Z') else '-')
            ax.axvline(x[-1]//3,color='black',alpha=.6,label='One-third capacity')
            ax.axvline(x[-1]//4,color='gray',linestyle=':',label='25% capacity')
            ax.set(xlabel='Models inspected',ylabel='Failures discovered',title=f'{cohort}: panel A, 10% observations')
            ax.legend(fontsize=8,loc='lower right')
    fig.tight_layout();fig.savefig(r/'allocation_control_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=True)
    for ax,variant in zip(axes,['A','A_on_B','B_refit']):
        x,a=curve('eligible',variant,'logistic_MZ');xx,b=curve('eligible',variant,'logistic_M')
        assert x==xx
        ax.step(x,[ai-bi for ai,bi in zip(a,b)],where='post',color='tab:purple')
        ax.axhline(0,color='gray');ax.axvline(x[-1]//3,color='black',alpha=.6)
        ax.axvline(x[-1]//4,color='gray',linestyle=':')
        ax.set(xlabel='Models inspected',title=variant)
        ax.set_ylim(-3.2,3.2)
    axes[0].set_ylabel('MZ minus M discoveries')
    fig.suptitle('Matched rich-control differences: complete source-normal curves')
    fig.tight_layout();fig.savefig(r/'matched_control_panel_curves.png',dpi=160);plt.close(fig)
    print('Saved allocation_control_curves.png and matched_control_panel_curves.png')

if __name__=='__main__':main()
