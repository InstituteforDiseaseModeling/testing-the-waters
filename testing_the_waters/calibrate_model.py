'''
This code is used to build the pars_* file containing parameter configuations
and corresponding rand_seed values.
'''

import os
import sciris as sc
import optuna as op
import numpy as np
import pandas as pd
import create_sim as cs
import covasim_schools as cvsch
import covasim as cv
import synthpops as sp
cv.check_save_version('1.7.6', folder='gitinfo', comments={'SynthPops':sc.gitinfo(sp.__file__)})

pop_size = 2.25e5
folder = 'v20201019'
children_equally_sus = False
alternate_symptomaticity = False
higher = False

# This is the data we're trying to fit
baseline_data = {
    'cases_begin':  75, # per 100k over 2-weeks
    'cases_end':    75, # per 100k over 2-weeks
    're':           1.0,
    'prevalence':   0.002,
    'yield':        0.024, # 2.4% positive
    'tests':        225,   # per 100k (per day) - 225 is 5000 tests in 2.23M pop per day
}

# King County statisits from https://coronavirus.wa.gov/what-you-need-know/covid-19-risk-assessment-dashboard
# Accessed 11/03/2020
higher_data = {
    'cases_begin':  104, # per 100k over 2-weeks, from DOH website
    'cases_end':    104, # per 100k over 2-weeks, from DOH website
    're':           1.0,
    'prevalence':   0.003, # KC image from Niket in #nowcasting-inside-voice dated 11/02 using data through 10/23
    'yield':        0.029, # 2.4% positive
    'tests':        225,   # per 100k (per day) - 225 is 5000 tests in 2.23M pop per day
}

to_fit = higher_data if higher else baseline_data

# And here are the weights for each element
weight = {
    'cases_begin':  1, # per 100k over 2-weeks
    'cases_end':    1, # per 100k over 2-weeks
    're':           1,
    'prevalence':   5,
    'yield':        1, # 2.4% positive
    'tests':        1,   # per 100k (per day) - 225 is 5000 tests in 2.23M pop per day
}

label = '_'.join([f'{k}={v}' for k,v in to_fit.items()])
name  = os.path.join(folder, f'pars_{label}_pop_size={int(pop_size)}')
if children_equally_sus:
    name += '_children_equally_sus'
if alternate_symptomaticity:
    name += '_alternate_symptomaticity'
if higher:
    name += '_higher'

storage   = f'sqlite:///{name}.db'
n_workers = 24
n_trials  = 20 # Each worker does n_trials
save_json = True

def scenario(es, ms, hs):
    return {
        'pk': None,
        'es': sc.dcp(es),
        'ms': sc.dcp(ms),
        'hs': sc.dcp(hs),
        'uv': None,
    }


def evaluate_sim(sim):
    first = sim.day('2020-11-02')
    last = sim.day('2021-01-31')

    ret = {
        'cases_begin': np.sum(sim.results['new_diagnoses'][(first-14):first]) * 1e5 / (sim.pars['pop_size'] * sim.pars['pop_scale']),
        'cases_end': np.sum(sim.results['new_diagnoses'][(last-14):last]) * 1e5 / (sim.pars['pop_size'] * sim.pars['pop_scale']),
        're': np.mean(sim.results['r_eff'][first:last]),
        'prevalence': np.mean(sim.results['prevalence'][first:last]),
        'yield': np.mean(sim.results['test_yield'][first:last]),
        'tests': np.mean(sim.results['new_tests'][first:last]) * 1e5 / (sim.pars['pop_size'] * sim.pars['pop_scale']),
    }

    mismatch = 0
    wsum = 0
    for key,true in to_fit.items():
        realized = ret[key]
        mismatch += weight[key] * np.abs(realized-true)/true
        wsum += weight[key]

    ret['mismatch'] = mismatch / wsum

    return ret # Weighted mean absolute error (scaled to true value)


def objective(trial, kind='default'):
    ''' Define the objective for Optuna '''
    pars = {}
    bounds = cs.define_pars(which='bounds', kind=kind)
    for key, bound in bounds.items():
        pars[key] = trial.suggest_uniform(key, *bound)
    pars['rand_seed'] = trial.number

    sim = cs.create_sim(pars, pop_size=pop_size, folder=folder, children_equally_sus=children_equally_sus, alternate_symptomaticity=alternate_symptomaticity)

    remote = {
        'start_day': '2020-11-02',
        'schedule': 'Remote',
        'screen_prob': 0,
        'test_prob': 0,
        'screen2pcr': 3, # Days from screening to receiving PCR results
        'trace_prob': 0,
        'quar_prob': 0,
        'ili_prob': 0,
        'beta_s': 0, # NOTE: No transmission in school layers
        'testing': None,
    }
    scen = scenario(es=remote, ms=remote, hs=remote)
    sm = cvsch.schools_manager(scen)
    sim['interventions'] += [sm]
    sim.run()

    mismatch = evaluate_sim(sim)['mismatch']
    return mismatch


def worker():
    ''' Run a single worker '''
    study = op.load_study(storage=storage, study_name=name)
    output = study.optimize(objective, n_trials=n_trials)
    return output


def run_workers():
    ''' Run multiple workers in parallel '''
    output = sc.parallelize(worker, n_workers)
    return output


def make_study(restart=True):
    ''' Make a study, deleting one if it already exists '''
    try:
        if restart:
            print(f'About to delete {storage}:{name}, you have 5 seconds to intervene!')
            sc.timedsleep(5.0)
            op.delete_study(storage=storage, study_name=name)
    except:
        pass

    output = op.create_study(storage=storage, study_name=name, load_if_exists=not(restart))
    return output


if __name__ == '__main__':
    t0 = sc.tic()
    make_study(restart=False)
    run_workers()
    study = op.load_study(storage=storage, study_name=name)
    best_pars = study.best_params
    T = sc.toc(t0, output=True)
    print(f'Output: {best_pars}, time: {T}')

    sc.heading('Loading data...')
    best = cs.define_pars('best')
    bounds = cs.define_pars('bounds')

    sc.heading('Making results structure...')
    results = []

    failed_trials = []
    for trial in study.trials:
        data = {'index': trial.number, 'mismatch': trial.value}
        for key, val in trial.params.items():
            data[key] = val
        if data['mismatch'] is None:
            failed_trials.append(data['index'])
        else:
            results.append(data)
    print(f'Processed {len(study.trials)} trials; {len(failed_trials)} failed')

    sc.heading('Making data structure...')
    keys = ['index', 'mismatch'] + list(best.keys())
    data = sc.objdict().make(keys=keys, vals=[])
    for i, r in enumerate(results):
        for key in keys:
            if key not in r:
                print(f'Warning! Key {key} is missing from trial {i}, replacing with default')
                r[key] = best[key]
            data[key].append(r[key])
    df = pd.DataFrame.from_dict(data)

    if save_json:
        order = np.argsort(df['mismatch'])
        json = []
        for o in order:
            row = df.iloc[o,:].to_dict()
            rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'), pars={})
            for key,val in row.items():
                rowdict['pars'][key] = val
            json.append(rowdict)
        sc.savejson(f'{name}.json', json, indent=2)
        saveobj = False
        if saveobj: # Smaller file, but less portable
            sc.saveobj(f'{name}.obj', json)
