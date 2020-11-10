'''
Compute the R0 value in different school types without interventions.
'''

import os
import numpy as np
import sciris as sc
import covasim as cv
import covasim_schools as cvsch
import create_sim as cs
from testing_scenarios import generate_scenarios

force_run = False
folder = 'v20201019'
par_inds = (0,30) # Select random seeds, can take a while with many

#%% Define the school seeding 'intervention'
class seed_schools(cv.Intervention):
    ''' Seed one infection in each school '''

    def __init__(self, n_infections=2, s_types=None, delay=0, choose_students=False, verbose=1, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so that intervention can be recreated
        self.n_infections = n_infections
        self.s_types = s_types if s_types else ['es', 'ms', 'hs']
        self.delay = delay
        self.choose_students = choose_students
        self.verbose = verbose
        self.school_ids = self._res([])
        self.seed_inds = self._res([])
        self.r0s = self._res(0.0)
        self.degree = self._res(0.0)
        self.numerators = self._res(0)
        self.denominators = self._res(0)
        return


    def _res(self, emptyobj):
        ''' Return a standard results dict -- like a defaultdict kind of '''
        return sc.objdict({k:sc.dcp(emptyobj) for k in self.s_types})


    def initialize(self, sim):
        ''' Find the schools and seed infections '''
        for st in self.s_types:
            self.school_ids[st] = sim.people.school_types[st]
            for sid in self.school_ids[st]:
                sch_uids = np.array(sim.people.schools[sid])
                if self.choose_students:
                    s_uids = cv.itruei(sim.people.student_flag, sch_uids)
                else:
                    s_uids = sch_uids
                choices = cv.choose(len(s_uids), self.n_infections)
                self.seed_inds[st] += s_uids[choices].tolist()

        return


    def apply(self, sim):
        if sim.t == self.delay: # Only infect on the first day (or after a delay)
            for st,inds in self.seed_inds.items():
                if len(inds):
                    sim.people.infect(inds=np.array(inds), layer=f'seed_infection_{st}')
                    for ind in inds:
                        sid = sim.people.school_id[ind]
                        sdf = sim.people.contacts[cvsch.int2key(sid)].to_df()
                        contacts = sdf.loc[ (sdf['p1']==ind) | (sdf['p2']==ind) ]
                        self.degree[st] += contacts.shape[0]
                if self.verbose:
                    print(f'Infected {len(inds)} people in school type {st} on day {sim.t}')

        if sim.t == sim.npts-1:
            self.tt = sim.make_transtree()
            self.tt.make_detailed(sim.people)
            for st in self.s_types:
                denominator = len(self.seed_inds[st])
                numerator = 0
                for ind in self.seed_inds[st]:
                    numerator +=  len(self.tt.targets[ind])
                self.numerators[st] = numerator
                self.denominators[st] = denominator
                if denominator:
                    self.r0s[st] = numerator/denominator
                self.r0s['overall'] = np.sum(self.numerators.values()) / np.sum(self.denominators.values())
            self.degree['overall'] = np.sum(self.degree.values()) / np.sum(self.denominators.values())
            for st in self.s_types:
                self.degree[st] /= self.denominators[st]
            sim.school_r0s = self.r0s
            sim.mean_degree = self.degree

        return


def run_sims():
    pop_size = 2.25e5
    calibfile = os.path.join(folder, 'pars_cases_begin=75_cases_end=75_re=1.0_prevalence=0.002_yield=0.024_tests=225_pop_size=225000.json')
    par_list = sc.loadjson(calibfile)[par_inds[0]:par_inds[1]]
    scen = generate_scenarios()['as_normal']

    for stype, cfg in scen.items():
        if cfg:
            cfg['start_day'] = '2020-09-07' # Move school start earlier

    # Configure and run the sims
    sims = []
    for eidx, entry in enumerate(par_list):
        par = sc.dcp(entry['pars'])
        par['rand_seed'] = int(entry['index'])

        # Clunky, but check that the population exists
        pop_seed = par['rand_seed'] % 5
        popfile = os.path.join(folder, 'inputs', f'kc_synthpops_clustered_{int(pop_size)}_withstaff_seed') + str(pop_seed) + '.ppl'
        if not os.path.exists(popfile):
            print(f'Population file {popfile} not found, recreating...')
            cvsch.make_population(pop_size=pop_size, rand_seed=par['rand_seed'], max_pop_seeds=5, popfile=popfile, do_save=True)

        par['pop_infected'] = 0 # Do NOT seed infections
        par['beta_layer'] = dict(h=0.0, s=0.0, w=0.0, c=0.0, l=0.0) # Turn off transmission in other layers, looking for in-school R0
        sim = cs.create_sim(par, pop_size=pop_size, folder=folder)

        delay = sim.day('2020-09-16') # Pick a Monday
        sim['interventions'] += [cvsch.schools_manager(scen), seed_schools(delay=delay, n_infections=1, choose_students=False)]
        sims.append(sim)

    msim = cv.MultiSim(sims)
    msim.run(keep_people=True)

    return msim


# Configuration
sc.heading('Configuring...')
T = sc.tic()

cachefn = os.path.join('v20201019', 'msims', f'R0_{par_inds[0]}-{par_inds[1]}.msim')
if force_run or cachefn is not None and not os.path.exists(cachefn):
    msim = run_sims()
    if cachefn is not None:
        print(f'Saving to {cachefn}')
        msim.save(cachefn, keep_people=False) # Don't think we need the people anymore
else:
    print(f'Loading from {cachefn}')
    msim = cv.MultiSim.load(cachefn)

# Results
res = {k:np.zeros(len(msim.sims)) for k in ['es', 'ms', 'hs', 'overall']}
deg = {k:np.zeros(len(msim.sims)) for k in ['es', 'ms', 'hs', 'overall']}
for s,sim in enumerate(msim.sims):
    for k in res.keys():
        res[k][s] = sim.school_r0s[k]
        deg[k][s] = sim.mean_degree[k]

for k in res.keys():
    mean = res[k].mean()
    std  = res[k].std()
    print(f'R0 for "{k}": {mean:0.2f} ± {std:0.2f}')

for k in res.keys():
    mean = deg[k].mean()
    std  = deg[k].std()
    print(f'Degree for "{k}": {mean:0.2f} ± {std:0.2f}')

print('Done.')
sc.toc(T)
