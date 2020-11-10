'''
Not a formal test -- check how many people are missing days.
'''

import os
import sciris as sc
import covasim as cv
import covasim_schools as cvsch
from testing_the_waters import create_sim as cs
from testing_the_waters.testing_scenarios import generate_scenarios

#%% Configuration
sc.heading('Configuring...')
T = sc.tic()

debug       = True # Verobisty and other settings
do_run      = True # Whether to rerun instead of load saved run
keep_people = False # Whether to keep people when running
parallelize = True # If running, whether to parallelize
do_save     = True # If rerunning, whether to save sims
do_plot     = True # Whether to plot results

n_seeds = 1 # Number of seeds to run each simulation with
rand_seed = 2346 # Overwrite the default random seed
bypass_popfile = 'explore_missed_days_small.ppl'
pop_size = int(20e3)

entry =   {
    "index": 376.0,
    "mismatch": 0.03221581045452142,
    "pars": {
      "pop_infected": 0,
      "change_beta": 0,
      "symp_prob": 0.0,
    }
  }
params = sc.dcp(entry['pars'])
if rand_seed is None:
    params['rand_seed'] = int(entry['index'])
else:
    params['rand_seed'] = rand_seed

# Ensure the population file exists
if not os.path.exists(bypass_popfile):
    print(f'Population file {bypass_popfile} not found, recreating...')
    cvsch.make_population(pop_size=pop_size, rand_seed=params['rand_seed'], max_pop_seeds=5, popfile=bypass_popfile, do_save=True)

origscen = generate_scenarios()['as_normal']
testings = {'No testing':
                None,
            'Antigen testing': { # Modify these values to explore different scenarios
                'start_date': '2020-10-26',
                'repeat': 14,
                'groups': ['students', 'teachers', 'staff'], # No students
                'coverage': 1,
                'is_antigen': True,
                'symp7d_sensitivity': 1.0, # https://www.fda.gov/media/141570/download
                'other_sensitivity': 1.0, # Modeling assumption
                'specificity': 1.0, # https://www.fda.gov/media/141570/download
                'PCR_followup_perc': 1.0,
                'PCR_followup_delay': 3.0,
            }}

t_keys = list(testings.keys())
n_testings = len(testings)
all_scens = sc.odict()
for tkey,testing in testings.items():
    scen = sc.dcp(origscen)
    for stype, spec in scen.items():
        if spec is not None:
            spec['testing'] = testing
    scen['es']['verbose'] = scen['ms']['verbose'] = scen['hs']['verbose'] = debug
    all_scens[tkey] = scen

# Create the sim
people = sc.loadobj(bypass_popfile)
base_sim = cs.create_sim(params, pop_size=pop_size, load_pop=False, people=people, verbose=0.1)


#%% Run the sims
sims = []
for key,scen in all_scens.items():
    for seed in range(n_seeds):
        sim = sc.dcp(base_sim)
        sim.set_seed(seed=sim['rand_seed'] + seed)
        sim.label = key
        sim['interventions'] += [cvsch.schools_manager(scen)]
        sims.append(sim)
msim_raw = cv.MultiSim(sims)
msim_raw.run()
msims = msim_raw.split(chunks=[n_seeds]*(len(sims)//n_seeds))
msims = list(msims)
res = sc.odict()
for msim in msims:
    msim.reduce()
    base = msim.base_sim
    sc.heading(base.label)
    print(base.school_results)
    res[base.label] = base.school_results


#%% Plotting
sc.heading('Plotting...')
if do_plot:
    msim_base = cv.MultiSim.merge(msims, base=True)
    msim_base.plot(to_plot='overview')
    plot_individual = False
    if plot_individual:
        msim_all = cv.MultiSim.merge(msims, base=False)
        msim_all.plot(to_plot='overview', color_by_sim=True, max_sims=2*n_seeds)


print('Done.')
sc.toc(T)