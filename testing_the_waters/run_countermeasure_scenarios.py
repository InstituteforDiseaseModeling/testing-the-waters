'''
Script to run several scenarios to determine why countermeasures work so well.

Note: not yet refactored for scalable runs.
'''

import os
import sciris as sc
import covasim as cv
import synthpops as sp
import covasim_schools as cvsch
import create_sim as cs
import testing_scenarios as t_s # From the local folder

cv.check_save_version('1.7.6', folder='gitinfo', comments={'SynthPops':sc.gitinfo(sp.__file__)})

par_inds = (0,30)
pop_size = 2.25e5
batch_size = 16

folder = 'v20201019'
stem = f'countermeasures_v2_{par_inds[0]}-{par_inds[1]}'
calibfile = os.path.join(folder, 'pars_cases_begin=75_cases_end=75_re=1.0_prevalence=0.002_yield=0.024_tests=225_pop_size=225000.json')


def baseline(sim, scen, test):
    # Modify scen
    for stype, spec in scen.items():
        if spec is not None:
            spec['testing'] = test # dcp probably not needed because deep copied in new_schools
    return scen

def no_NPI_reduction(sim, scen, test):
    # Modify scen
    for stype, spec in scen.items():
        if spec is not None:
            if spec['beta_s'] > 0:
                spec['beta_s'] = 1.5 # Restore to pre-NPI level
    return scen

def no_screening(sim, scen, test):
    # Modify scen
    for stype, spec in scen.items():
        if spec is not None:
            spec['screen_prob'] = 0
    return scen

def no_tracing(sim, scen, test):
    # Modify scen
    for stype, spec in scen.items():
        if spec is not None:
            spec['trace_prob'] = 0
    return scen


if __name__ == '__main__':
    scenarios = t_s.generate_scenarios()
    # Select a subset, if desired:
    #scenarios = {k:v for k,v in scenarios.items() if k in ['with_countermeasures', 'all_hybrid', 'k5', 'all_remote']}
    #scenarios = {k:v for k,v in scenarios.items() if k in ['with_countermeasures']}

    testing = t_s.generate_testing()
    # Select a subset, if desired:
    #testing = {k:v for k,v in testing.items() if k in ['None', 'PCR every 2w', 'Antigen every 1w teach&staff, PCR f/u', 'Antigen every 2w, PCR f/u']}
    #testing = {k:v for k,v in testing.items() if k in ['None', 'Antigen every 2w, PCR f/u']}

    sensitivity = {
        # Baseline
        'baseline': [baseline],

        # For K-5 in particular, masks could be challenging - what if we remove the 25% NPI boost?
        # --> Change beta in the scenario
        'NPI': [baseline, no_NPI_reduction],

        # screen_prob was 90%
        # --> Remove symptom screening
        'screening': [baseline, no_screening],

        # trace_prob was 75%
        # --> Remove tracing
        'tracing': [baseline, no_tracing],

        'NPI_screening': [baseline, no_NPI_reduction, no_screening],
        'NPI_tracing': [baseline, no_NPI_reduction, no_tracing],
        'screening_tracing': [baseline, no_screening, no_tracing],
        'NPI_screening_tracing': [baseline, no_NPI_reduction, no_screening, no_tracing],
    }

    # Select a subset, if desired:
    #sensitivity = {k:v for k,v in sensitivity.items() if k in ['baseline']}

    par_list = sc.loadjson(calibfile)[par_inds[0]:par_inds[1]]

    sims = []
    msims = []
    tot = len(scenarios) * len(testing) * len(par_list) * len(sensitivity)
    proc = 0

    # Save time by pre-generating the base simulations
    base_sims = []
    for eidx, entry in enumerate(par_list):
        par = sc.dcp(entry['pars'])
        par['rand_seed'] = int(entry['index'])
        base_sim = cs.create_sim(par, pop_size=pop_size, folder=folder)
        base_sim.dynamic_par = par
        base_sims.append(base_sim)

    for senskey, builders in sensitivity.items():
        print(f'Beginning {senskey}')
        for eidx, base_sim in enumerate(base_sims):
            for sidx, (skey, scen) in enumerate(scenarios.items()):
                for tidx, (tkey, test) in enumerate(testing.items()):
                    sim = base_sim.copy()

                    sim.label = f'{skey} + {tkey}'
                    sim.key1 = skey
                    sim.key2 = tkey
                    sim.key3 = senskey
                    sim.eidx = eidx
                    sim.scen = scen
                    sim.tscen = test
                    sim.dynamic_par = par

                    # Call the function to build the sensitivity analysis
                    modscen = sc.dcp(scen)
                    for builder in builders:
                        modscen = builder(sim, modscen, sc.dcp(test))


                    sm = cvsch.schools_manager(modscen)
                    sim['interventions'] += [sm]

                    sims.append(sim)
                    proc += 1

                    if len(sims) == batch_size or proc == tot or (tidx == len(testing)-1 and sidx == len(scenarios)-1 and eidx == len(par_list)-1):
                        print(f'Running sims {proc-len(sims)}-{proc-1} of {tot}')
                        msim = cv.MultiSim(sims)
                        msims.append(msim)
                        msim.run(reseed=False, par_args={'ncpus': 16}, noise=0.0, keep_people=False)
                        sims = []

        fn = os.path.join(folder, 'msims', f'{stem}_{senskey}.msim')
        print(f'*** Saving to {fn} after completing {senskey}')
        sims_this_scenario = [s for msim in msims for s in msim.sims if s.key3 == senskey]
        msim = cv.MultiSim(sims_this_scenario)
        cv.save(fn, msim)

    msim = cv.MultiSim.merge(msims)
    msim.base_sim = [] # Save disk space
    fn = os.path.join(folder, 'msims', f'{stem}.msim')
    print(f'Saving msims to {fn}')
    cv.save(fn, msim)
