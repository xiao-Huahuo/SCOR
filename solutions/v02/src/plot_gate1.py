"""Render exportable scientific figures from audited Gate1 CSV/JSON results."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from pilot_utils import atomic_json,file_digest,read_json


def plot(directory:Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter
    summary=read_json(directory/'result_summary.json')
    with (directory/'paired_seeds.csv').open(encoding='utf-8',newline='') as stream:
        paired=list(csv.DictReader(stream))
    valid=[row for row in paired if row['status']=='valid']
    n=len(valid)
    if n!=summary['valid_seeds'] or len(paired)!=len(summary['planned_seeds']):
        raise ValueError('Figure sample counts differ from audited result')
    plt.rcParams.update({'font.family':'serif','font.serif':['STIX Two Text','STIXGeneral','DejaVu Serif'],
                         'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    primary=summary['primary'];epsilon=primary.get('epsilon_D')
    if n:
        with (directory/'score_curves.csv').open(encoding='utf-8',newline='') as stream:
            curves=list(csv.DictReader(stream))
        steps=[0,8,16,24,32]
        seeds=[int(row['seed']) for row in valid]
        if {int(row['seed']) for row in curves}!=set(seeds):raise ValueError('Curve seed set differs')
        matrix=[];accuracy={'P_clean':[],'P_corrupt':[],'P_repeat':[]}
        for i,seed in enumerate(seeds):
            rows=sorted([row for row in curves if int(row['seed'])==seed],key=lambda row:int(row['step']))
            if [int(row['step']) for row in rows]!=steps:raise ValueError('Missing curve points')
            values=[float(row['corrected_D']) for row in rows];matrix.append(values)
            final=next(row for row in valid if int(row['seed'])==seed)
            if (not all(math.isfinite(v) for v in values) or
                    not math.isclose(values[-1]-epsilon,float(final['Q_s']),abs_tol=1e-12)):
                raise ValueError('Curve endpoint differs from the paired Q_s')
            axes[0].plot(steps,values,color=plt.cm.tab10(i),alpha=.65,lw=1,label=str(seed))
            for key in accuracy:accuracy[key].append([float(row[key]) for row in rows])
        axes[0].plot(steps,np.mean(matrix,axis=0),color='black',lw=2,label='Seed mean')
        if epsilon is not None:axes[0].axhline(epsilon,color='black',ls='--',lw=1,label='Frozen tolerance')
        axes[0].axhline(0,color='0.7',lw=.7)
        axes[0].set(xlabel='Training step',ylabel='D_score − clean-repeat floor (nat/token)',title='Fixed-probe score curves')
        axes[0].set_xticks(steps);axes[0].legend(fontsize=8)
        endpoint=[float(row['Q_s']) for row in valid]
        axes[1].scatter(range(1,n+1),endpoint,color='tab:blue',label='Paired seed')
        mean=primary['mean'];lower=primary.get('lower');upper=primary.get('upper')
        if not math.isclose(float(np.mean(endpoint)),mean,abs_tol=1e-12):
            raise ValueError('Figure endpoint mean differs from the audited primary endpoint')
        if lower is not None and upper is not None:
            axes[1].errorbar(n+1,mean,yerr=[[mean-lower],[upper-mean]],fmt='o',color='black',capsize=4,
                             label='Mean and 95% paired t interval')
        else:axes[1].scatter([n+1],[mean],color='black',label='Mean; interval unavailable')
        axes[1].axhline(0,color='black',ls='--',lw=1,label='Frozen decision boundary')
        axes[1].set_xticks(range(1,n+2),[str(seed)[-2:] for seed in seeds]+['Mean'])
        axes[1].set(xlabel='Training seed (last two digits)',ylabel='Q_s = corrected D_score − ε_D (nat/token)',title='Primary endpoint at T=32')
        axes[1].legend(fontsize=8)
        for key,label,color in [('P_clean','Clean-fixed','tab:blue'),('P_corrupt','Corrupt-fixed','tab:orange'),
                                ('P_repeat','Clean-repeat','0.5')]:
            values=np.asarray(accuracy[key])
            for row in values:axes[2].plot(steps,row,color=color,alpha=.2,lw=.8)
            axes[2].plot(steps,values.mean(axis=0),color=color,lw=2,label=label)
        axes[2].set(xlabel='Training step',ylabel='Explicit-answer accuracy',ylim=(0,1),title='Held-out accuracy (auxiliary)')
        axes[2].set_xticks(steps);axes[2].yaxis.set_major_formatter(PercentFormatter(1));axes[2].legend(fontsize=8)
    else:
        for axis in axes:
            axis.text(.5,.5,'No eligible intervention targets\nNo paired effect estimate',ha='center',va='center',transform=axis.transAxes)
            axis.set_axis_off()
    fig.suptitle(f'Gate1 | {n}/{len(paired)} eligible independent seed blocks | fixed future training data',fontsize=13)
    fig.savefig(directory/'gate1_results.png',dpi=220)
    fig.savefig(directory/'gate1_results.svg')
    plt.close(fig)
    sources=['result_summary.json','paired_seeds.csv']+(['score_curves.csv'] if n else [])
    receipt={'source_sha256':file_digest(Path(__file__)),
             'data_sha256':{name:file_digest(directory/name) for name in sources},
             'outputs':{name:file_digest(directory/name) for name in ['gate1_results.png','gate1_results.svg']},
             'sample_unit':'independent paired training seed; cases/tokens are not extra repeats',
             'interval_note':'95% paired Student-t endpoint interval; small-sample assumption; curves are descriptive'}
    atomic_json(directory/'figure_manifest.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    args=parser.parse_args();plot(args.directory)
