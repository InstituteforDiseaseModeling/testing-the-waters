'''
Run simulations to check the model calibration. Produces Figure 9.
'''

import os
import psutil
import multiprocessing as mp
import numpy as np
import covasim as cv
import create_sim as cs
import sciris as sc
import matplotlib.pyplot as plt
import covasim_schools as cvsch
import testing_scenarios as t_s
import synthpops as sp
from calibrate_model import evaluate_sim
from pathlib import Path
cv.check_save_version('1.7.6', folder='gitinfo', comments={'SynthPops':sc.gitinfo(sp.__file__)})


#%% Configuration

do_run = False
parallel = True
do_plot = True
do_plot_overview = False


children_equally_sus = False
alternate_symptomaticity = False
assert( not children_equally_sus or not alternate_symptomaticity )

cpu_thresh = 0.75 # Don't use more than this amount of available CPUs, if number of CPUs is not set
mem_thresh = 0.75 # Don't use more than this amount of available RAM, if number of CPUs is not set
n_cpus = None
par_inds = (0,30) # First and last parameters to run
pop_size = 2.25e5
batch_size = 30
verbose = 0.1 # Print progress this fraction of simulated days (1 = every day, 0.1 = every 10 days, 0 = no output)

folder = 'v20201019'

# Choose which pars file to load
if children_equally_sus:
    stem = f'calib_cheqsu_{par_inds[0]}-{par_inds[1]}'
    calibfile = os.path.join(folder, 'pars_cases_begin=75_cases_end=75_re=1.0_prevalence=0.002_yield=0.024_tests=225_pop_size=225000_children_equally_sus.json')
elif alternate_symptomaticity:
    stem = f'calib_altsymp_{par_inds[0]}-{par_inds[1]}'
    calibfile = os.path.join(folder, 'pars_cases_begin=75_cases_end=75_re=1.0_prevalence=0.002_yield=0.024_tests=225_pop_size=225000_alternate_symptomaticity.json')
else:
    stem = f'calib_{par_inds[0]}-{par_inds[1]}'
    calibfile = os.path.join(folder, 'pars_cases_begin=75_cases_end=75_re=1.0_prevalence=0.002_yield=0.024_tests=225_pop_size=225000.json')

imgdir = os.path.join(folder, 'img_'+stem)
Path(imgdir).mkdir(parents=True, exist_ok=True)


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



#%% Running
def create_run_sim(sconf, n_sims):
    ''' Create and run the actual simulations '''
    print(f'Creating and running sim {sconf.count} of {n_sims}...')
    T = sc.tic()

    sim = cs.create_sim(sconf.pars, pop_size=pop_size, folder=folder, children_equally_sus=children_equally_sus, alternate_symptomaticity=alternate_symptomaticity)

    # Modify scen with test
    this_scen = sc.dcp(sconf.scen)
    for stype, spec in this_scen.items():
        if spec is not None:
            spec['testing'] = sc.dcp(sconf.test)
            spec['beta_s'] = 1.5

    # Add information to the sim
    sim.label = f'{sconf.skey} + {sconf.tkey}'
    sim.key1 = sconf.skey
    sim.key2 = sconf.tkey
    sim.scen = this_scen
    sim.tscen = sconf.test
    sim.dynamic_par = sconf.pars

    # Create the schools intervention
    sm = cvsch.schools_manager(this_scen)
    sim['interventions'] += [sm]

    # Run the simulation
    sim.run(verbose=verbose)
    sim.shrink() # Do not keep people after run
    sc.toc(T)
    return sim


if __name__ == '__main__':

    TT = sc.tic()


    #%% Running

    if do_run:

        # Choose what to run
        sc.heading('Configuring sims...')
        skey = 'all_remote'
        scen = t_s.generate_scenarios()[skey]
        tkey = 'None'
        test = t_s.generate_testing()[tkey]
        par_list = sc.loadjson(calibfile)[par_inds[0]:par_inds[1]]
        sim_configs = []

        # Create the configurations
        for eidx, entry in enumerate(par_list):
            sconf = sc.objdict()
            pars = sc.dcp(entry['pars'])
            pars['rand_seed'] = int(entry['index'])
            sconf.count = eidx
            sconf.pars = pars
            sconf.skey = skey
            sconf.tkey = tkey
            sconf.scen = scen
            sconf.test = test
            sim_configs.append(sconf)


        # Run it
        sc.heading('Running sims...')

        if parallel:
            sims = sc.parallelize(create_run_sim, iterarg=sim_configs, kwargs=dict(n_sims=len(sim_configs)), ncpus=n_cpus)
        else:
            sims = []
            for sconf in sim_configs:
                sim = create_run_sim(sconf, n_sims=len(sim_configs))
                sims.append(sim)

        sc.heading('Saving all sims...')
        filename = os.path.join(folder, 'sims', f'{stem}.sims')
        cv.save(filename, sims)
        print(f'Saved {filename}; done.')

    else:
        # Load the results
        sims = cv.load(os.path.join(folder, 'sims', f'{stem}.sims'))



    #%% Plotting

    if do_plot:

        sc.heading('Plotting...')

        ms = cv.MultiSim(sims)

        for s,sim in enumerate(sims):
            print(f'Statistics for simulation {s} ({sim.label})')
            stats = evaluate_sim(sim)
            sc.pp(stats)

        ms.reduce()
        if do_plot_overview:
            ms.plot(to_plot='overview') # Nice overview plot, if desired

        to_plot = sc.odict({
                'New Infections per 225k': [
                    'new_infections',
                ],
                'New Diagnoses per 225k': [
                    'new_diagnoses',
                ],
                'Test Yield': [
                    'test_yield',
                ],
                'Effective Reproduction Number': [
                    'r_eff',
                ],
                'New Tests per 225k': [
                    'new_tests',
                ],
                'Prevalence': [
                    'prevalence',
                ],
            })

        axes_layout = [3,2]
        f, axv = plt.subplots(*axes_layout, figsize=(12,8))
        chs = sc.odict({
            'new_infections': { 'title': 'New Infections per 100k', 'ref':None },
            'new_diagnoses': { 'title': 'New Diagnoses over 14d per 100k', 'ref': 75 },
            'test_yield': { 'title': 'Test Yield', 'ref': 0.022 },
            'r_eff': { 'title': 'Reproduction Number', 'ref': 1.0 },
            'new_tests': { 'title': 'New Tests per 100k', 'ref': 225 },
            'prevalence': { 'title': 'Prevalence', 'ref': 0.002 },
        })

        for count,ch,info in chs.enumitems():
            ri,ci = np.unravel_index(count, axes_layout)

            rvec = []
            for sim in sims:
                r = sim.results[ch].values
                ps = 100000 / sim.pars['pop_size']*sim.pars['pop_scale']
                if ch in ['new_diagnoses']:
                    r *= 14 * ps
                if ch in ['new_infections', 'new_tests']:
                    r *= ps
                rvec.append(r)

            ax = axv[ri,ci]
            med = np.median(rvec, axis=0)
            sd = np.std(rvec, axis=0)
            ax.fill_between(sim.results['date'], med+2*sd, med-2*sd, color='lightgray')
            ax.plot(sim.results['date'], med+2*sd, color='gray', lw=1)
            ax.plot(sim.results['date'], med-2*sd, color='gray', lw=1)
            ax.plot(sim.results['date'], med, color='k', lw=2)
            if 'ref' in info and info['ref'] is not None:
                ax.axhline(y=info['ref'], color='r', ls='--', lw=2)
            ax.set_title(info['title'])

        f.tight_layout()
        fn = os.path.join(imgdir, f'calib.png')
        print(f'Saving figure to {fn}')
        cv.savefig(fn, dpi=300)


    sc.toc(TT)
    print('Done.')
