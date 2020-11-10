'''
Script compares PCR testing of various return delays to antigen testing, either with default parameters or with perfect sensitivity and specificity
'''

import os
import numpy as np
import pandas as pd
import seaborn as sns
import covasim as cv
import create_sim as cs
import sciris as sc
import matplotlib.pyplot as plt
import matplotlib as mplt
from school_intervention import new_schools
from testing_scenarios import generate_scenarios, generate_testing
import synthpops as sp
cv.check_save_version('1.7.2', comments={'SynthPops':sc.gitinfo(sp.__file__)})

# Global plotting styles
font_size = 16
font_style = 'Roboto Condensed'
mplt.rcParams['font.size'] = font_size
mplt.rcParams['font.family'] = font_style

do_run = True

par_inds = (0,20) # First and last parameters to run
pop_size = 2.25e5 # 1e5 2.25e4 2.25e5
batch_size = 16

folder = 'v20201019'
imgdir = os.path.join(folder, 'img')
stem = f'pcr_vs_ag_{par_inds[0]}-{par_inds[1]}'
calibfile = os.path.join(folder, 'pars_cases_begin=75_cases_end=75_re=1.0_prevalence=0.002_yield=0.024_tests=225_pop_size=225000.json')

if __name__ == '__main__':
    scenarios = generate_scenarios()
    scenarios = {k:v for k,v in scenarios.items() if k in ['as_normal']}

    tests = generate_testing()
    pcr = tests['PCR every 2w']
    ag = tests['Antigen every 2w, PCR f/u']

    testing = {}
    for pcr_delay in range(4):
        t = sc.dcp(pcr)
        t[0]['delay'] = pcr_delay
        testing[pcr_delay] = t
    testing['Antigen'] = ag

    perfect_ag = sc.dcp(ag)
    perfect_ag[0]['symp7d_sensitivity'] = 1.0
    perfect_ag[0]['other_sensitivity'] = 1.0
    perfect_ag[0]['specificity'] = 1.0

    testing['Perfect antigen'] = perfect_ag

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

                    sim.label = f'{skey} + {tkey}'
                    sim.key1 = skey
                    sim.key2 = tkey
                    sim.eidx = eidx
                    sim.scen = scen
                    sim.tscen = test
                    sim.dynamic_par = par

                    # modify scen with test
                    this_scen = sc.dcp(scen)
                    for stype, spec in this_scen.items():
                        if spec is not None:
                            spec['testing'] = sc.dcp(test) # dcp probably not needed because deep copied in new_schools

                    ns = new_schools(this_scen)
                    sim['interventions'] += [ns]
                    sims.append(sim)
                    proc += 1

                    if len(sims) == batch_size or proc == tot:# or (tidx == len(testing)-1 and eidx == len(par_list)-1):
                        print(f'running sims {proc-len(sims)}-{proc-1} of {tot}')
                        msim = cv.MultiSim(sims)
                        msims.append(msim)
                        msim.run(reseed=False, par_args={'ncpus': 16}, noise=0.0, keep_people=False)
                        sims = []

            print(f'*** saving after completing {skey}')
            sims_this_scenario = [s for msim in msims for s in msim.sims if s.key1 == skey]
            msim = cv.MultiSim(sims_this_scenario)

        # Save results
        msim = cv.MultiSim.merge(msims)
        msim.save(os.path.join(folder, 'msims', f'{stem}.msim'), keep_people=False)
    else:
        # Load results
        msim = cv.MultiSim.load(os.path.join(folder, 'msims', f'{stem}.msim'))

    # GENERATE PLOTS
    grp_dict = {'Students': ['students'], 'Teachers & Staff': ['teachers', 'staff']}
    groups = ['students', 'teachers', 'staff']
    results = []

    for sim in msim.sims:
        first_school_day = sim.day('2020-11-02')
        last_school_day = sim.day('2021-01-31')
        ret = {
            'key1': sim.key1,
            'key2': sim.key2,
            'eidx': sim.eidx,
        }

        exposed = {k:0 for k in grp_dict.keys()}
        count = {k:0 for k in grp_dict.keys()}

        first_date = '2020-11-02'
        first = sim.day(first_date)
        last_date = '2021-01-31'
        last = sim.day(last_date)

        for sid,stats in sim.school_stats.items():
            if stats['type'] not in ['es', 'ms', 'hs']:
                continue

            for gkey, grps in grp_dict.items():
                for grp in grps:
                    exposed[gkey] += np.sum(stats['newly_exposed'][grp])
                    count[gkey] += stats['num'][grp]

        # Deciding between district and school perspective here
        for gkey in grp_dict.keys():
            ret[f'attackrate_{gkey}'] = 100*exposed[gkey] / count[gkey] #np.mean(attackrate[gkey])

        results.append(ret)

    df = pd.DataFrame(results)

    df.replace({'key2':
        {
            0: 'PCR results same day',
            1: 'PCR results next day',
            2: 'PCR results in two days',
            3: 'PCR results in three days',
            4: 'PCR results in four days',
            'Antigen': 'Antigen w/PCR follow-up',
        }
    }, inplace=True)

    blues = plt.cm.get_cmap('Blues')
    reds = plt.cm.get_cmap('Reds')
    cdict = {
        'PCR results same day': blues(4/5),
        'PCR results next day': blues(3/5),
        'PCR results in two days': blues(2/5),
        'PCR results in three days': blues(1/5),
        'PCR results in four days': blues(0/5),
        'Antigen w/PCR follow-up': reds(1/2),
        'Perfect antigen': reds(2/2),
    }

    d = pd.melt(df, id_vars=['key1', 'key2'], value_vars=[f'attackrate_{gkey}' for gkey in grp_dict.keys()], var_name='Group', value_name='Cum Inc (%)')
    d.replace( {'Group': {f'attackrate_{gkey}':gkey for gkey in grp_dict.keys()}}, inplace=True)

    f, ax = plt.subplots(figsize=(9,6))
    sns.barplot(data=d, x='Group', y='Cum Inc (%)', hue='key2', ax=ax, hue_order=['PCR results in three days', 'PCR results in two days', 'PCR results next day', 'PCR results same day', 'Antigen w/PCR follow-up', 'Perfect antigen'], palette=cdict)
    ax.set_ylabel("3-Month Attack Rate (%)")
    ax.set_xlabel("")
    ax.legend(title=None)
    plt.tight_layout()
    cv.savefig(os.path.join(imgdir, f'PCR_vs_Ag_3mAttackRate.png'), dpi=300)
