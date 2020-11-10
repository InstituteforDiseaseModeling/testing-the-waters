'''
Produce Figure 3 in the "Testing the Waters" paper that explores how the impact of screening on the percent of schools that may have an infectious individul present on the first day varies with the number of days between screening and the first in-person day.
'''

import os
import pandas as pd
import seaborn as sns
import covasim as cv
import create_sim as cs
import sciris as sc
import matplotlib.pyplot as plt
import matplotlib as mplt
import covasim_schools as cvsch
import testing_scenarios as t_s # From the local folder
from pathlib import Path
import synthpops as sp

cv.check_save_version('1.7.6', folder='gitinfo', comments={'SynthPops':sc.gitinfo(sp.__file__)})

# Global plotting styles
font_size = 18
font_style = 'Roboto Condensed'
mplt.rcParams['font.size'] = font_size
mplt.rcParams['font.family'] = font_style

do_run = True

par_inds = (0,30) # First and last parameters to run
pop_size = 2.25e5 # 1e5 2.25e4 2.25e5
batch_size = 16

folder = 'v20201019'
stem = f'pcr_days_sweep_{par_inds[0]}-{par_inds[1]}'
imgdir = os.path.join(folder, 'img_'+stem)
Path(imgdir).mkdir(parents=True, exist_ok=True)

calibfile = os.path.join(folder, 'pars_cases_begin=75_cases_end=75_re=1.0_prevalence=0.002_yield=0.024_tests=225_pop_size=225000.json')

if __name__ == '__main__':
    scenarios = t_s.generate_scenarios()
    scenarios = {k:v for k,v in scenarios.items() if k in ['with_countermeasures']}

    test = t_s.generate_testing()['PCR 1w prior']
    test[0]['delay'] = 0 # Otherwise "same day" will not work.

    testing = {}
    # Sweep over these days on which to conduct the screening
    for start_date in ['None', '2020-10-26', '2020-10-27', '2020-10-28', '2020-10-29', '2020-10-30', '2020-10-31', '2020-11-01', '2020-11-02']:
        t = sc.dcp(test)
        if start_date == 'None':
            t[0]['start_date'] = '2022-01-01' # Move into the deep future for the "None" scenario
        else:
            t[0]['start_date'] = start_date

        testing[start_date] = t

    par_list = sc.loadjson(calibfile)[par_inds[0]:par_inds[1]]

    if do_run:
        # Run the simulations
        sims = []
        msims = []
        tot = len(scenarios) * len(testing) * len(par_list)
        proc = 0
        for skey, scen in scenarios.items():
            for tidx, (tkey, test) in enumerate(testing.items()):
                for eidx, entry in enumerate(par_list):
                    par = sc.dcp(entry['pars'])
                    par['rand_seed'] = int(entry['index'])
                    sim = cs.create_sim(par, pop_size=pop_size, folder=folder)

                    # Modify scen with test
                    this_scen = sc.dcp(scen)
                    for stype, spec in this_scen.items():
                        if spec is not None:
                            spec['testing'] = sc.dcp(test) # dcp probably not needed because deep copied in new_schools

                    ns = cvsch.schools_manager(this_scen)
                    sim['interventions'] += [ns]

                    sim.label = f'{skey} + {tkey}'
                    sim.key1 = skey
                    sim.key2 = tkey
                    sim.scen = scen
                    sim.tscen = test
                    sim.dynamic_par = par
                    sims.append(sim)

                    proc += 1

                    if len(sims) == batch_size or proc == tot:# or (tidx == len(testing)-1 and eidx == len(par_list)-1):
                        print(f'Running sims {proc-len(sims)}-{proc-1} of {tot}')
                        msim = cv.MultiSim(sims)
                        msims.append(msim)
                        msim.run(reseed=False, par_args={'ncpus': 16}, noise=0.0, keep_people=False)
                        sims = []

            print(f'*** Saving after completing {skey}')
            sims_this_scenario = [s for msim in msims for s in msim.sims if s.key1 == skey]
            msim = cv.MultiSim(sims_this_scenario)

        # Save results
        msim = cv.MultiSim.merge(msims)
        fn = os.path.join(folder, 'msims', f'{stem}.msim')
        print(f'Saving to {fn}')
        msim.save(fn, keep_people=False)
    else:
        # Load results
        fn = os.path.join(folder, 'msims', f'{stem}.msim')
        print(f'Loading from {fn}')
        msim = cv.MultiSim.load(fn)


    # GENERATE PLOTS
    byschool = []
    groups = ['students', 'teachers', 'staff']
    for sim in msim.sims:
        # Note: The objective function has recently changed, so mismatch will not match!
        first_school_day = sim.day('2020-11-02')
        last_school_day = sim.day('2021-01-31')
        for sid,stats in sim.school_stats.items():
            if stats['type'] not in ['es', 'ms', 'hs']:
                continue

            inf_at_sch = stats['infectious_stay_at_school'] # stats['infectious_arrive_at_school'] stats['infectious_stay_at_school']

            byschool.append({
                'type': stats['type'],
                'key1': sim.key1, # Filtered to just one scenario (key1)
                'key2': sim.key2,
                'n_students': stats['num']['students'], #sum([stats['num'][g] for g in groups]),
                'd1 infectious': sum([inf_at_sch[g][first_school_day] for g in groups]),
                'd1 bool': sum([inf_at_sch[g][first_school_day] for g in groups]) > 0,
            })


    d = pd.DataFrame(byschool)
    colors = plt.cm.get_cmap('cool')
    fig, ax = plt.subplots(figsize=(12,8))
    N = len(d.groupby('key2'))
    for i, (key, dat) in enumerate(d.groupby('key2')):
        c = 'k' if key == 'None' else colors(i/N)
        sns.regplot(data=dat, x='n_students', y='d1 bool', logistic=True, y_jitter=0.03, scatter_kws={'s':5}, label=key, ax=ax, ci=None, scatter=False, color=c)
    ax.set_xlabel('School size (students)')
    ax.set_ylabel('Infection on First Day (%)')
    yt = [0.0, 0.25, 0.50, 0.75]
    ax.set_yticks(yt)
    ax.set_yticklabels([int(100*t) for t in yt])
    ax.grid(True)

    lmap = {
        'None':'No PCR testing',
        '2020-10-26': 'One week prior',
        '2020-10-27': 'Six days prior',
        '2020-10-28': 'Five days prior',
        '2020-10-29': 'Four days prior',
        '2020-10-30': 'Three days prior',
        '2020-10-31': 'Two days prior',
        '2020-11-01': 'One day prior',
        '2020-11-02': 'On the first day of school'
    }

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[-1]] + handles[:-1]
    labels = [lmap[labels[-1]]] + [lmap[l] for l in labels[:-1]]

    ax.legend(handles, labels)
    plt.tight_layout()
    cv.savefig(os.path.join(imgdir, 'PCR_Days_Sweep.png'), dpi=300)

    fig.tight_layout()

