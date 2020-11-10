'''
Not a formal test, but a script for running all scenarios. Based on test_schools.py.
'''

import os
import numpy as np
import pylab as pl
import sciris as sc
import seaborn as sns
import covasim as cv
import covasim_schools as cvsch
from testing_the_waters import create_sim as cs
from testing_the_waters.testing_scenarios import generate_scenarios, generate_testing

#%% Configuration
sc.heading('Configuring...')
T = sc.tic()

debug       = True # Verobisty and other settings
bypass      = True # Whether to use a small population size
do_run      = True # Whether to rerun instead of load saved run
keep_people = False # Whether to keep people when running
parallelize = True # If running, whether to parallelize
do_save     = True # If rerunning, whether to save sims
do_plot     = True # Whether to plot results

rand_seed = 2346 # Overwrite the default random seed
folder = '../testing_the_waters/v20201019'
bypass_popfile = 'explore_scenarios_small.ppl'
sims_file = 'explore_scenarios.sims'
pop_size = int(20e3) if bypass else int(2.25e5)
calibfile = os.path.join(folder, 'pars_cases_begin=75_cases_end=75_re=1.0_prevalence=0.002_yield=0.024_tests=225_pop_size=225000.json')

try:
    entry = sc.loadjson(calibfile)[1]
except Exception as E:
    entry =   {
        "index": 376.0,
        "mismatch": 0.03221581045452142,
        "pars": {
          "pop_infected": 242.11186358945181,
          "change_beta": 0.5313884845187986,
          "symp_prob": 0.08250498122080606
        }
      }
    print(f'Warning: could not load calibration file "{calibfile}" due to "{str(E)}", using hard-coded parameters')
params = sc.dcp(entry['pars'])
if rand_seed is None:
    params['rand_seed'] = int(entry['index'])
else:
    params['rand_seed'] = rand_seed

# Ensure the population file exists
if bypass and not os.path.exists(bypass_popfile):
    print(f'Population file {bypass_popfile} not found, recreating...')
    cvsch.make_population(pop_size=pop_size, rand_seed=params['rand_seed'], max_pop_seeds=5, popfile=bypass_popfile, do_save=True)

# Create the scenarios
divider = ' -- '
def joinkeys(skey, tkey):
    ''' Turn scenario and testing keys into a single key '''
    return divider.join([skey, tkey])

def splitkey(key):
    ''' Oppostite of joinkeys() '''
    return key.split(divider)

scens = generate_scenarios()
testings = generate_testing()
s_keys = list(scens.keys())
t_keys = list(testings.keys())
n_scens = len(scens)
n_testings = len(testings)
all_scens = sc.odict()
all_keys = []
for skey,origscen in scens.items():
    for tkey,testing in testings.items():
        scen = sc.dcp(origscen)
        for stype, spec in scen.items():
            if spec is not None:
                spec['testing'] = testing
        scen['es']['verbose'] = scen['ms']['verbose'] = scen['hs']['verbose'] = debug
        all_scens[joinkeys(skey, tkey)] = scen
        all_keys.append([skey, tkey])

# Create the sim
if bypass: # BYPASS option -- create small population on the fly
    people = sc.loadobj(bypass_popfile)
    base_sim = cs.create_sim(params, pop_size=pop_size, load_pop=False, people=people, verbose=0.1)
else: # Otherwise, load full population from disk
    base_sim = cs.create_sim(params, pop_size=pop_size, folder=folder, verbose=0.1)


#%% Run the sims
def run_sim(scen):
    ''' Run a single sim '''
    sim = sc.dcp(base_sim)
    sm = cvsch.schools_manager(scen)
    sim['interventions'] += [sm]
    sim.run(keep_people=keep_people)
    return sim

if do_run:
    if parallelize:
        sc.heading('Running in parallel...')
        raw_sims = sc.parallelize(run_sim, all_scens.values())
        sims = sc.odict({k:scen for k,scen in zip(all_scens.keys(), raw_sims)})
    else:
        sc.heading('Running in serial...')
        sims = sc.odict()
        for k,scen in all_scens:
            sims[k] = run_sim(scen)
    if do_save:
        sc.saveobj(sims_file, sims)

else:
    sc.heading('Loading from disk...')
    sims = sc.loadobj(sims_file)


#%% Analysis
sc.heading('Analyzing...')
res = sc.objdict()

def arr():
    return np.zeros((n_scens, n_testings))

res.cum_infections = arr()
res.cum_tests = arr()

shared_keys = sims[0].school_results.shared_keys
subkeys = ['students', 'teachers+staff']
for key in shared_keys:
    for k2 in subkeys:
        res[f'{key}_{k2}'] = arr()
res.PCR_tests = arr()
res.Antigen_tests = arr()

for s,skey in enumerate(s_keys):
    for t,tkey in enumerate(t_keys):
        sim = sims[joinkeys(skey, tkey)]
        res.cum_infections[s,t] = sim.results['cum_infections'][-1]
        res.cum_tests[s,t] = sim.results['cum_tests'][-1]

        for key in shared_keys:
            for k2 in subkeys:
                res[f'{key}_{k2}'][s,t] = sim.school_results[key][k2]

        res.PCR_tests[s,t] = sim.school_results.n_tested.PCR
        res.Antigen_tests[s,t] = sim.school_results.n_tested.Antigen

n_res = len(res)


#%% Plotting
sc.heading('Plotting...')

fig,axs = pl.subplots(4,5)
flataxs = axs.flatten()
δ = 0.03
pl.subplots_adjust(left=δ, right=1-δ, bottom=δ, top=1-δ, hspace=0.25, wspace=0.25)

strings = []
strings.append('Scenario definitions:')
for s,skey in enumerate(s_keys):
    strings.append(f'  S{s} -- {skey}')
strings.append('Testing definitions:')
for t,tkey in enumerate(t_keys):
    strings.append(f'  T{t} -- {tkey}')

for r,key,thisres in res.enumitems():
    ax = flataxs[r]
    pl.sca(ax)
    sns.heatmap(res[key], annot=True, fmt='0.0f', annot_kws={'fontsize':8})
    ax.set_title(key)
    ax.set_xticks(np.arange(n_testings)+0.5)
    ax.set_xticklabels(np.arange(n_testings))
    ax.set_yticks(np.arange(n_scens)+0.5)
    ax.set_yticklabels(np.arange(n_scens))
    ax.set_xlabel('Testing scenario')
    ax.set_ylabel('Classroom scenario')

labelax = flataxs[-1]
for s,string in enumerate(strings):
    x = 0.1
    y = 0.9*(1-(s/len(strings)))
    labelax.text(x, y, string)

cv.maximize(fig)

print('Done.')
sc.toc(T)