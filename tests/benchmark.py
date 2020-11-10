'''
Benchmark the intervention
'''

import os
import sciris as sc
import covasim as cv
import synthpops as sp
import covasim_schools as cvsch
from testing_the_waters import create_sim as cs
from testing_the_waters.testing_scenarios import generate_scenarios, generate_testing
from testing_the_waters.calibrate_model import evaluate_sim

cv.check_save_version('1.7.2', folder='gitinfo', comments={'SynthPops':sc.gitinfo(sp.__file__)})

debug = False
use_intervention = True
# NOTE: The following may be bypassed below by hard-coded pop_size and folder
folder = '../testing_the_waters/v20201019'
pop_size = 2.25e5 # 1e5 2.25e4 2.25e5
calibfile = os.path.join(folder, 'pars_cases_begin=75_cases_end=75_re=1.0_prevalence=0.002_yield=0.024_tests=225_pop_size=225000.json')

def scenario(es, ms, hs):
    return {
        'pk': None,
        'es': sc.dcp(es),
        'ms': sc.dcp(ms),
        'hs': sc.dcp(hs),
        'uv': None,
    }

def benchmark_schools():

    entry = sc.loadjson(calibfile)[0]
    params = sc.dcp(entry['pars'])
    params['rand_seed'] = int(entry['index'])

    scen = generate_scenarios()['with_countermeasures']
    testing = generate_testing()['Antigen every 2w, PCR f/u']
    #testing[0]['delay'] = 0
    for stype, spec in scen.items():
        if spec is not None:
            spec['testing'] = testing
    scen['testing'] = testing
    scen['es']['verbose'] = scen['ms']['verbose'] = scen['hs']['verbose'] = debug

    sim = cs.create_sim(params, pop_size=pop_size, folder=folder, verbose=0.1)

    if use_intervention:
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

    #sim.save('test.sim')
    #cv.savefig('sim.png')

    return sim


if __name__ == '__main__':

    to_profile = 'stats_update'

    func_options = dict(
        step = cv.Sim.step,
        school_update = cvsch.School.update,
        stats_update = cvsch.SchoolStats.update,
        )

    sc.profile(run=benchmark_schools, follow=func_options[to_profile])
