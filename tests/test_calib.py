'''
Run a calibration-like simulation to see if the resulting objective function
value matches that from calibration.

NB: skipped by default since the full population files etc. won't exist on GitHub.
'''

import os
import sciris as sc
import covasim as cv
import covasim_schools as cvsch
from testing_the_waters import create_sim as cs
from testing_the_waters.testing_scenarios import generate_scenarios, generate_testing
from testing_the_waters.calibrate_model import evaluate_sim
import pytest

debug = False
folder = '../testing_the_waters/v20201016_225k'
pop_size = 2.25e5 # 1e5 2.25e4 2.25e5
calibfile = os.path.join(folder, 'pars_cases_begin=75_cases_end=75_re=1.0_prevalence=0.002_yield=0.024_tests=225_pop_size=225000.json')


@pytest.mark.skip # Will not be able to find the right files automatically
def test_calib():

    entry = sc.loadjson(calibfile)[0]
    params = sc.dcp(entry['pars'])
    params['rand_seed'] = int(entry['index'])

    scen = generate_scenarios()['all_remote']
    testing = generate_testing()['None']
    #testing[0]['delay'] = 0
    for stype, spec in scen.items():
        if spec is not None:
            spec['testing'] = testing
    scen['testing'] = testing
    scen['es']['verbose'] = scen['ms']['verbose'] = scen['hs']['verbose'] = debug

    sim = cs.create_sim(params, pop_size=pop_size, folder=folder)

    sm = cvsch.schools_manager(scen)
    sim['interventions'] += [sm]

    sim.run(keep_people=debug)

    stats = evaluate_sim(sim)
    print(stats)

    if debug:
        sim.plot(to_plot='overview')
        #t = sim.make_transtree()
    else:
        sim.plot()

    cv.savefig('sim.png')

    return sim


if __name__ == '__main__':

    sim = test_calib()
