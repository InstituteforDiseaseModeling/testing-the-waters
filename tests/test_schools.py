'''
Script for quick and dirty single run of a school-based intervention
'''

import os
import sciris as sc
import covasim_schools as cvsch
from testing_the_waters import create_sim as cs
from testing_the_waters.testing_scenarios import generate_scenarios, generate_testing
from testing_the_waters.calibrate_model import evaluate_sim

debug = True
# NOTE: The following may be bypassed below by hard-coded pop_size and folder
bypass = True
folder = '../testing_the_waters/v20201019'
pop_size = 2.25e5 # 1e5 2.25e4 2.25e5
calibfile = os.path.join(folder, 'pars_cases_begin=75_cases_end=75_re=1.0_prevalence=0.002_yield=0.024_tests=225_pop_size=225000.json')


def test_schools(do_plot=False):

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
    params['rand_seed'] = int(entry['index'])

    scen = generate_scenarios()['all_hybrid']
    testing = generate_testing()['Antigen every 1w, PCR f/u']
    #testing[0]['repeat'] = 1
    for stype, spec in scen.items():
        if spec is not None:
            spec['testing'] = testing
    scen['es']['verbose'] = scen['ms']['verbose'] = scen['hs']['verbose'] = debug

    if bypass: # BYPASS option -- create small population on the fly
        sim = cs.create_sim(params, pop_size=int(20e3), load_pop=False)
    else: # Otherwise, load full population from disk
        sim = cs.create_sim(params, pop_size=pop_size, folder=folder, verbose=0.1)

    sm = cvsch.schools_manager(scen)
    sim['interventions'] += [sm]

    sim.run(keep_people=debug)

    stats = evaluate_sim(sim)
    print(stats)

    if do_plot:
        if debug:
            sim.plot(to_plot='overview')
            #t = sim.make_transtree()
        else:
            sim.plot()

    #sim.save('test.sim')
    #cv.savefig('sim.png')

    return sim


if __name__ == '__main__':
    sim = test_schools(do_plot=True)
