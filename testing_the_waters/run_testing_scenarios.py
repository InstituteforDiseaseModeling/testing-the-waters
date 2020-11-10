'''
This is the main script used for commissioning the different testing scenarios
(defined in testing_scenarios.py) for the paper results. Run this script first
(preferably on an HPC!), then run the different plotting scripts. Each sim takes
about 1 minute to run. With full settings, there are about 1300 scripts to run;
the test run uses 8.
'''

import os
import psutil
import multiprocessing as mp
import covasim as cv
import create_sim as cs
import sciris as sc
import synthpops as sp
import covasim_schools as cvsch
import testing_scenarios as t_s # From the local folder


#%% Settings
sc.heading('Setting parameters...')

# Check that versions are correct
cv.check_save_version('1.7.6', folder='gitinfo', comments={'SynthPops':sc.gitinfo(sp.__file__)})

test_run = False # Whether to do a small test run, or the full results: changes the number of runs and scenarios -- 1 for testing, or 30 for full results
parallel = True # Only switch to False for debugging
par_inds = [0,2] if test_run else [0,30]
pop_size = 2.25e5
skip_screening = False # Set True for the no-screening variant
save_each_sim = False # Save each sim separately instead of all together
n_cpus = None # Manually set the number of CPUs -- otherwise calculated automatically
cpu_thresh = 0.75 # Don't use more than this amount of available CPUs, if number of CPUs is not set
mem_thresh = 0.75 # Don't use more than this amount of available RAM, if number of CPUs is not set
verbose = 0.1 if test_run else 0.0 # Print progress this fraction of simulated days (1 = every day, 0.1 = every 10 days, 0 = no output)

folder = 'v20201019'
stem = f'final_20201026_{par_inds[0]}-{par_inds[1]}'
calibfile = os.path.join(folder, 'pars_cases_begin=75_cases_end=75_re=1.0_prevalence=0.002_yield=0.024_tests=225_pop_size=225000.json')

# For a test run only use a subset of scenarios
scenarios = t_s.generate_scenarios() # Can potentially select a subset of scenarios
testing = t_s.generate_testing() # Potentially select a subset of testing
if test_run:
     scenarios = {k:v for k,v in scenarios.items() if k in ['as_normal', 'all_hybrid']}
     testing = {k:v for k,v in testing.items() if k in ['None', 'Antigen every 1w, PCR f/u']}

par_list = sc.loadjson(calibfile)[par_inds[0]:par_inds[1]]


sc.heading('Choosing correct number of CPUs...')
if n_cpus is None:
    cpu_limit = int(mp.cpu_count()*cpu_thresh) # Don't use more than 75% of available CPUs
    ram_available = psutil.virtual_memory().available/1e9
    ram_required = 1.5*pop_size/2.25e5 # Roughly 1.5 GB per 225e3 people
    ram_limit = int(ram_available/ram_required*mem_thresh)
    n_cpus = min(cpu_limit, ram_limit)
    print(f'{n_cpus} CPUs are being used due to a CPU limit of {cpu_limit} and estimated RAM limit of {ram_limit}')
else:
    print(f'Using user-specified {n_cpus} CPUs')



#%% Configuration
sc.heading('Creating sim configurations...')
sim_configs = []
count = -1
for skey, base_scen in scenarios.items():
    for tidx, (tkey, test) in enumerate(testing.items()):
        for eidx, entry in enumerate(par_list):
            count += 1

            pars = sc.dcp(entry['pars'])
            pars['rand_seed'] = int(entry['index'])

            sconf = sc.objdict(count=count, pars=pars, pop_size=pop_size, folder=folder)

            # Modify base_scen with testing intervention
            this_scen = sc.dcp(base_scen)
            for stype, spec in this_scen.items():
                if spec is not None:
                    spec['testing'] = sc.dcp(test) # dcp probably not needed because deep copied in new_schools
                    if skip_screening:
                        spec['screen_prob'] = 0

            sm = cvsch.schools_manager(this_scen)

            sconf.update(dict(
                label = f'{skey} + {tkey}',
                skey = skey,
                tkey = tkey,
                eidx = eidx,
                test = test,
                this_scen = this_scen,
                sm = sm,
            ))

            sim_configs.append(sconf)
print(f'Done: {len(sim_configs)} configurations created')



#%% Running
def create_run_sim(sconf, n_sims):
    ''' Create and run the actual simulations '''
    print(f'Creating and running sim {sconf.count} of {n_sims}...')
    T = sc.tic()
    sim = cs.create_sim(sconf.pars, pop_size=sconf.pop_size, folder=sconf.folder)
    sim.count = sconf.count
    sim.label = sconf.label
    sim.key1 = sconf.skey
    sim.key2 = sconf.tkey
    sim.eidx = sconf.eidx
    sim.tscen = sconf.test
    sim.scen = sconf.this_scen # After modification with testing above
    sim.dynamic_par = sconf.pars
    sim['interventions'] += [sconf.sm]
    sim.run(verbose=verbose)
    sim.shrink() # Do not keep people after run
    if save_each_sim:
        filename = os.path.join(folder, 'sims', f'sim{sconf.count}_{skey}.sim')
        sim.save(filename)
        print(f'Saved {filename}')
    sc.toc(T)
    return sim


# Windows requires a main block for running in parallel
if __name__ == '__main__':

    sc.heading('Running sims...')

    TT = sc.tic()

    if parallel:
        sims = sc.parallelize(create_run_sim, iterarg=sim_configs, kwargs=dict(n_sims=len(sim_configs)), ncpus=n_cpus)
    else:
        sims = []
        for sconf in sim_configs:
            sim = create_run_sim(sconf, n_sims=len(sim_configs))
            sims.append(sim)

    if not save_each_sim:
       sc.heading('Saving all sims...')
       filename = os.path.join(folder, 'sims', f'{stem}.sims')
       cv.save(filename, sims)
       print(f'Saved {filename}')

    print('Done.')
    sc.toc(TT)
