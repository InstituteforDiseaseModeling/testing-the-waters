'''
Main file implementing school-based interventions.  The user interface is handled
by the schools_manager() intervention. This primarily uses the School class, of
which there is one instance per school. SchoolTesting orchestrates testing within
a school, while SchoolStats records results. The remaining functions are contact
managers, which handle different cohorting options (and school days).
'''

import covasim as cv
import numpy as np
import sciris as sc

__all__ = ['schools_manager', 'SchoolScenario', 'School', 'SchoolTesting', 'SchoolStats', 'int2key', 'groups']


# Define the groups that statistics are computed for
groups = ['students', 'teachers', 'staff']

def int2key(x):
    ''' Convert an school ID integer to a dict key, e.g. 5 -> 's5' '''
    return 's' + str(x)


class schools_manager(cv.Intervention):
    '''
    This is the front end to the Schools class intervention.  Not much here,
    it's a sub-class of covasim's Intervention class.  The one piece of computation
    done here is to split the original school 's' network into individual schools.
    (Each school was already a separate component, but the code figures out which
    component goes with which school and extracts the subgraph.)

    The only input argument (aside from standard Intervention ones) is the scenario
    dict, which has one entry for each of the five school types, e.g.

        scenario = {
            'pk': None,
            'es': es_pars,
            'ms': ms_pars,
            'hs': hs_pars,
            'uv': None,
        }

    each of which has the following example structure:

            scen_pars = {
                'start_day': '2020-11-02',
                'schedule': 'Full',
                'screen_prob': 0.9,
                'test_prob': 0.5, # Amongst those who screen positive
                'screen2pcr': 3, # Days from screening to receiving PCR results
                'trace_prob': 0.75, # Fraction of newly diagnosed index cases who are traced
                'quar_prob': 0.75, # Of those reached by contact tracing, this fraction will quarantine
                'ili_prob': 0.002, # Daily ili probability equates to about 10% incidence over the first 3 months of school
                'beta_s': 0.75 * base_beta_s, # 25% reduction due to NPI
                'testing': test_pars,
            }

    The testing parameters object is a dict (or list of dicts), with the following
    structure. These coordinate testing interventions in schools:

            test_pars = {
                'start_date': '2020-10-26',
                'repeat': 7,
                'groups': ['teachers', 'staff'],
                'coverage': 1,
                'sensitivity': 1,
                'delay': 1,
            }
    '''

    def __init__(self, scenario, shrink=True, **kwargs):
        super().__init__(**kwargs) # Initialize the Intervention object
        self._store_args() # Store the input arguments so that intervention can be recreated

        # Store arguments
        self.shrink = shrink # Whether to shrink the object at the end by removing the schools
        self.scenario = SchoolScenario(scenario)
        self.schools = []
        return

    def initialize(self, sim):
        # Create schools, stealing 's' edges into the School class instances upon *initialize*
        self.school_types = sim.people.school_types # Dict with keys of school types (e.g. 'es') and values of list of school ids (e.g. [1,5])

        sdf = sim.people.contacts['s'].to_df()
        sim.school_stats = {}

        for school_type, scids in self.school_types.items():
            for school_id in scids:
                uids = sim.people.schools[school_id] # Dict with keys of school_id and values of uids in that school

                if self.scenario[school_type] is not None:

                    stats = {
                        'type':         school_type,
                        'scenario':     self.scenario[school_type],
                    }
                    sim.school_stats[int2key(school_id)] = stats

                    # Extract 's'-layer associated with this school
                    rows = (sdf['p1'].isin(uids)) | (sdf['p2'].isin(uids))
                    s_subset = sdf.loc[ rows ]
                    sdf = sdf.loc[ ~rows ] # Remove rows from the 's' contacts
                    this_school_layer = cv.Layer().from_df(s_subset)

                    sch = School(sim, school_id, school_type, uids, this_school_layer, **self.scenario[school_type])
                    self.schools.append(sch)

                    # Configure the new layer
                    sim['beta_layer'][sch.sid] = self.scenario[school_type]['beta_s']
                    sim['iso_factor'][sch.sid] = sim['iso_factor']['s']
                    sim['quar_factor'][sch.sid] = sim['quar_factor']['s']

        # Delete remaining entries in sim.people.contacts['s'], these were associated with schools that will not open, e.g. pk and uv
        sim.people.contacts['s'] = cv.Layer()

        self.initialized = True

    def apply(self, sim):
        for school in self.schools:
            layer = school.update(sim)
            sim.people.contacts[school.sid] = layer

        if sim.t == sim.npts-1:
            # Only needed on final time step:
            for school in self.schools:
                sim.school_stats[school.sid].update(school.get_stats())
            self.gather_stats(sim)
            if self.shrink:
                self.schools = [] # Huge space savings if user saves this simulation due to python junk collection


    def gather_stats(self, sim):
        ''' Gather statistics from individual schools into one object '''

        def standard_res():
            ''' Define a standard results object '''
            return sc.objdict({'students':0, 'teachers':0, 'staff':0, 'teachers+staff':0, 'all':0})

        # Keys to copy over/sum over from each school
        shared_keys = ['num', 'infectious_stay_at_school', 'newly_exposed', 'scheduled', 'in_person']

        res = sc.objdict()
        res.shared_keys = shared_keys # Store this here for ease of later use
        res.n_schools = len(self.schools)
        res.n_school_days = sim.school_stats[self.schools[0].sid]['num_school_days'] # Should be the same for all schools
        res.n_tested = sc.objdict({'PCR': 0, 'Antigen': 0})
        for key in shared_keys:
            res[key] = standard_res()

        # Count the stats
        for school in self.schools:
            stats = sim.school_stats[school.sid]
            for group in groups:
                for key in shared_keys:
                    res[key][group] += np.sum(stats[key][group]) # Main results
            for key in res.n_tested.keys(): # Count tests
                res.n_tested[key] += stats['n_tested'][key]

        # Compute combined keys
        for key in shared_keys:
            res[key]['teachers+staff'] = res[key]['teachers'] + res[key]['staff']
            res[key]['all'] = res[key]['teachers+staff'] + res[key]['students']

        # Save to the sim and return
        sim.school_results = res
        return res



class SchoolScenario(cv.FlexDict):
    '''
    Lightweight class for ensuring the scenarios are specified correctly. See
    schools_manager() for structure and definition.

    Example:

        SchoolScenario(param_dict)
    '''
    def __init__(self, scendict):

        # Define the required keys
        school_type_keys = [
            'pk', # Preschool/kindergarten
            'es', # Elementary
            'ms', # Middle
            'hs', # High
            'uv', # University
            ]

        scen_keys = [
            'start_day', # Day the scenario starts
            'schedule', # Full, hybrid, or remote scheduling
            'screen_prob', # Probability of a screening test
            'test_prob', # Probability of a confirmatory PCR test
            'screen2pcr', # Days from screening to receiving PCR results
            'trace_prob', # Fraction of newly diagnosed index cases who are traced
            'quar_prob', # Of those reached by contact tracing, this fraction will quarantine
            'ili_prob', # Prevalence of influenza-like symptoms
            'beta_s', # Reduction in beta due to NPI
            'testing', # Defined below
        ]

        optional_scen_keys = ['verbose']

        # Keys used by both testing interventions
        shared_test_keys = [
            'start_date', # Date testing program starts
            'repeat', # How frequently testing is repeated
            'groups', # Which out of students, teachers, staff test
            'coverage', # Proportion tested
            'is_antigen', # Whether or not it's an antigen test
        ]

        # PCR-specific keys
        pcr_test_keys = [
            'sensitivity', # Test sensitivity
            'delay', # Days until results are returned
        ]

        # Antigen-specific keys
        antigen_test_keys = [
            'symp7d_sensitivity', # Test sensitivity, within first 7 days of symptoms
            'other_sensitivity', # Test sensitivity otherwise
            'specificity', # Test specificity (false positive rate)
            'PCR_followup_perc', # Proportion who receive PCR follow-up
            'PCR_followup_delay', # Delay until PCR follow-up
        ]

        # Validate scenario
        if set(school_type_keys) != set(scendict.keys()):
            errormsg = f'Mismatch between expected school types ({school_type_keys}) and supplied ({set(scendict.keys())})'
            raise ValueError(errormsg)
        for st_key,scen_pars in scendict.items(): # Loop over school types
            if scen_pars is None:
                self[st_key] = None # Just skip
            else:
                self[st_key] = cv.FlexDict()
                kwarg_keys = set(scen_pars.keys())
                missing = set(scen_keys) - kwarg_keys
                extra = kwarg_keys - set(scen_keys + optional_scen_keys)
                if extra or missing: # Loop over scenario parameters
                    errormsg = f'In your scenario definition, you are missing keys "{missing}" and have extra keys "{extra}"'
                    raise ValueError(errormsg)
                else:
                    for key in scen_keys:
                        self[st_key][key] = scen_pars[key]

                # Validate testing
                self[st_key]['testing'] = sc.promotetolist(self[st_key]['testing']) # Ensure it's a list for iteration
                for e,entry in enumerate(self[st_key]['testing']):
                    entry_keys = set(entry.keys())
                    if 'is_antigen' in entry and entry['is_antigen']: # It's an antigen test
                        test_keys = antigen_test_keys + shared_test_keys
                    else:
                        test_keys = pcr_test_keys + shared_test_keys
                        entry['is_antigen'] = False # By default, assume not an antigen test
                    if entry_keys != set(test_keys):
                        missing = set(test_keys) - entry_keys
                        extra = entry_keys - set(test_keys)
                        errormsg = f'In your testing definition (position {e}), you are missing keys "{missing}" and have extra keys "{extra}"'
                        raise ValueError(errormsg)

        return



class School(sc.prettyobj):
    ''' Represents a single school; handle the layer updates and coordinating testing and other tasks '''

    def __init__(self, sim, school_id, school_type, uids, layer,
                start_day, screen_prob, screen2pcr, test_prob, trace_prob, quar_prob,
                schedule, beta_s, ili_prob, testing, verbose=False, **kwargs):
        '''
        Initialize the School

        sim          (covasim Sim)  : Pointer to the simulation object
        school_id    (int)          : ID of this school
        school_type  (str)          : Type of this school in pk, es, ms, hs, uv
        uids         (array)        : Array of ids of individuals associated with this school
        layer        (Layer)        : The fragment of the original 's' network associated with this school
        start_day    (str)          : Opening day for school
        screen_prob  (float)        : Coverage of screening
        test_prob    (float)        : Probability of PCR testing on screen +
        screen2pcr   (int)          : Days between positive screening receiving PCR results, for those testing
        trace_prob   (float)        : Probability of tracing from PCR+
        quar_prob    (float)        : Probability school contacts quarantine on trace
        schedule     (str)          : Full, Hybrid, or Remote
        beta_s       (float)        : beta for this school
        ili_prob     (float)        : Daily probability of ILI
        testing      (struct)       : List of dictionaries of parameters for SchoolTesting
        '''

        self.sid = int2key(school_id) # Convert to an string
        self.stype = school_type
        self.uids = np.array(uids)
        self.start_day = sim.day(start_day)
        self.screen_prob = screen_prob
        self.screen2pcr = screen2pcr
        self.test_prob = test_prob
        self.trace_prob = trace_prob
        self.quar_prob = quar_prob
        self.schedule = schedule
        self.beta_s = beta_s # Not currently used here, but rather in the school_intervention
        self.ili_prob = ili_prob
        self.verbose = verbose

        # TODO: these flags should have been arrays in population.py, but weren't.  Convert here for performance.
        sim.people.student_flag = np.array(sim.people.student_flag, dtype=bool)
        sim.people.teacher_flag = np.array(sim.people.teacher_flag, dtype=bool)
        sim.people.staff_flag = np.array(sim.people.staff_flag, dtype=bool)

        self.is_open = False # Schools start closed

        self.uids_at_home = {} # Dict from uid to release date

        if self.schedule.lower() == 'hybrid':
            self.ct_mgr = HybridContactManager(sim, self.uids, layer)
        elif self.schedule.lower() == 'full':
            self.ct_mgr = FullTimeContactManager(sim, self.uids, layer)
        elif self.schedule.lower() == 'remote':
            self.ct_mgr = RemoteContactManager(sim, self.uids, layer)
        else:
            print(f'Warning: Unrecognized schedule ({self.schedule}) passed to School class.')

        self.stats = SchoolStats(self, sim)
        self.testing = SchoolTesting(self, testing, sim)
        self.empty_layer = cv.Layer() # Cache an empty layer
        return


    def screen(self, sim):
        ''' Screen those individuals who are arriving at school '''

        # Inclusion criteria: diagnosed or symptomatic but not recovered and not dead
        inds_to_screen = cv.binomial_filter(self.screen_prob, self.uids_arriving_at_school)
        if len(inds_to_screen) == 0:
            return np.empty(0)

        ppl = sim.people
        symp = ppl.symptomatic[inds_to_screen]

        rec_or_dead = np.logical_or(ppl.recovered[inds_to_screen], ppl.dead[inds_to_screen])
        screen_pos = np.logical_and(symp, ~rec_or_dead)
        screen_pos_uids = cv.itrue(screen_pos, inds_to_screen)

        # Add in screen positives from ILI amongst those who were screened negative
        if self.ili_prob is not None and self.ili_prob > 0:
            screen_neg_uids = cv.ifalse(screen_pos, inds_to_screen)
            n_ili = np.random.binomial(len(screen_neg_uids), self.ili_prob) # Poisson
            if n_ili > 0:
                ili_pos_uids = np.random.choice(screen_neg_uids, n_ili, replace=False)
                screen_pos_uids = np.concatenate((screen_pos_uids, ili_pos_uids))

        return screen_pos_uids


    def update(self, sim):
        ''' Process the day, return the school layer '''

        # Even if a school is not yet open, consider testing in the population
        ids_to_iso = self.testing.update(sim)
        self.uids_at_home.update(ids_to_iso)

        # Look for newly diagnosed people (by PCR)
        newly_dx_inds = cv.itrue(sim.people.date_diagnosed[self.uids] == sim.t, self.uids) # Diagnosed this time step, time to trace

        if self.verbose and len(newly_dx_inds)>0: print(sim.t, f'School {self.sid} has {len(newly_dx_inds)} newly diagnosed: {newly_dx_inds}', [sim.people.date_exposed[u] for u in newly_dx_inds], 'recovering', [sim.people.date_recovered[u] for u in newly_dx_inds])

        # Isolate newly diagnosed individuals - could happen before school starts
        for uid in newly_dx_inds:
            self.uids_at_home[uid] = sim.t + sim.pars['quar_period'] # Can come back after quarantine period

        # If any individuals are done with quarantine, return them to school
        self.uids_at_home = {uid:date for uid,date in self.uids_at_home.items() if date >= sim.t} # >= or =?

        # Check if school is open
        if not self.is_open:
            if sim.t == self.start_day:
                if self.verbose:
                    print(sim.t, self.sid, f'School {self.sid} is opening today with {len(self.uids_at_home)} at home: {self.uids_at_home}')

                    infectious_uids = cv.itrue(sim.people.infectious[self.uids], self.uids)
                    print(sim.t, self.sid, 'Infectious:', len(cv.true(sim.people.infectious[self.uids])) * sim.rescale_vec[sim.t], len(infectious_uids) )
                    print(sim.t, self.sid, 'Iuids:', infectious_uids)
                    print(sim.t, self.sid, 'Itime:', [sim.people.date_exposed[u] for u in infectious_uids])
                self.is_open = True
            else:
                # CLOSED SCHOOLS DO NOT PASS THIS POINT!
                return self.empty_layer

        date = sim.date(sim.t)
        self.scheduled_uids = self.ct_mgr.begin_day(date) # Call at the beginning of the update

        # Quarantine contacts of newly diagnosed individuals - # TODO: Schedule in a delay
        if len(newly_dx_inds) > 0:
            # Identify school contacts to quarantine
            uids_to_trace = np.array(cv.binomial_filter(self.trace_prob, newly_dx_inds), dtype='int64') # This has to be an int64 (the default type)
            uids_reached_by_tracing = self.ct_mgr.find_contacts(uids_to_trace) # Assume all contacts of traced individuals will quarantine
            uids_to_quar = cv.binomial_filter(self.quar_prob, uids_reached_by_tracing)

            # Quarantine school contacts
            for uid in uids_to_quar:
                self.uids_at_home[uid] = sim.t + sim.pars['quar_period'] # Can come back after quarantine period

            # N.B. Not intentionally testing those in quarantine other than what covasim already does

        # Determine who will arrive at school (used in screen() and stats.update())
        self.uids_arriving_at_school = np.setdiff1d(self.scheduled_uids, np.fromiter(self.uids_at_home.keys(), dtype=int))

        # Perform symptom screening
        screen_pos_ids = self.screen(sim)

        if len(screen_pos_ids) > 0:
            # Perform follow-up testing on some
            uids_to_test = cv.binomial_filter(self.test_prob, screen_pos_ids)
            sim.people.test(uids_to_test, test_delay=self.screen2pcr)
            #sim.results['new_tests'][t] += len(uids_to_test)
            self.testing.n_tested['PCR'] += len(uids_to_test) # Ugly, move all testing in to the SchoolTesting class!

            # Send the screen positives home - quar_period if no PCR and otherwise the time to the PCR
            for uid in uids_to_test:
                self.uids_at_home[uid] = sim.t + self.screen2pcr # Can come back after PCR results are in

            for uid in np.setdiff1d(screen_pos_ids, uids_to_test):
                self.uids_at_home[uid] = sim.t + sim.pars['quar_period'] # Can come back after quarantine period


        # Determine (for tracking, mostly) who has arrived at school and passed symptom screening
        uids_at_home_array = np.fromiter(self.uids_at_home.keys(), dtype=int)
        self.uids_passed_screening = np.setdiff1d(self.scheduled_uids, uids_at_home_array)

        # Remove individuals at home from the network
        self.ct_mgr.remove_individuals(uids_at_home_array)

        self.stats.update(sim, self) # Supply the schools object to the stats update object so it doesn't need recursion
        # if sim.t == sim.npts-1:
        #     self.stats.finalize()

        # Return what is left of the layer
        return self.ct_mgr.get_layer()

    def get_stats(self):
        ''' Return a dictionary of statistics '''

        return self.stats.get(self)



class SchoolTesting(sc.prettyobj):
    '''
    Conduct testing in school students and staff.

    N.B. screening with follow-up testing is handled in the School class.
    '''

    def __init__(self, school, testing, sim):
        ''' Initialize testing. '''

        self.school = school
        self.testing = [] if testing is None else sc.dcp(testing)

        for test in self.testing:
            if 'is_antigen' not in test:
                test['is_antigen'] = False

        self.n_tested = { 'PCR': 0, 'Antigen': 0 }

        for test in self.testing:
            # Determine from test start_day and repeat which sim times to test on
            start_t = sim.day(test['start_date'])
            if test['repeat'] == None:
                # Easy - one time test
                test['t_vec'] = [start_t]
            else:
                test['t_vec'] = list(range(start_t, sim.pars['n_days'], test['repeat']))

            # Determine uids to include
            uids = []
            ppl = sim.people
            if 'students' in test['groups']:
                uids.append( cv.itruei(ppl.student_flag, self.school.uids) )
            if 'staff' in test['groups']:
                uids.append( cv.itruei(ppl.staff_flag, self.school.uids) )
            if 'teachers' in test['groups']:
                uids.append( cv.itruei(ppl.teacher_flag, self.school.uids) )
            test['uids'] = np.concatenate(uids)


    def antigen_test(self, inds, sym7d_sens=1.0, other_sens=1.0, specificity=1, loss_prob=0.0, sim=None):
        '''
        Adapted from the test() method on sim.people to do antigen testing. Main change is that sensitivity is now broken into those symptomatic in the past week and others.

        Args:
            inds: indices of who to test
            sym7d_sens (float): probability of a true positive in a recently symptomatic individual (7d)
            other_sens (float): probability of a true positive in others
            loss_prob (float): probability of loss to follow-up
            delay (int): number of days before test results are ready
        '''

        ppl = sim.people
        t = sim.t

        inds = np.unique(inds)
        # Antigen tests don't count towards stats (yet)
        #ppl.tested[inds] = True
        #ppl.date_tested[inds] = t # Only keep the last time they tested
        #ppl.date_results[inds] = t + delay # Keep date when next results will be returned

        is_infectious_not_dx = cv.itruei(ppl.infectious * ~ppl.diagnosed, inds)
        symp = cv.itruei(ppl.symptomatic, is_infectious_not_dx)
        recently_symp_inds = symp[t-ppl.date_symptomatic[symp] < 7]

        other_inds = np.setdiff1d(is_infectious_not_dx, recently_symp_inds)

        is_inf_pos = np.concatenate((
            cv.binomial_filter(sym7d_sens, recently_symp_inds), # Higher sensitivity for <7 days
            cv.binomial_filter(other_sens, other_inds)          # Lower sensitivity of otheres
        ))

        not_lost           = cv.n_binomial(1.0-loss_prob, len(is_inf_pos))
        true_positive_uids = is_inf_pos[not_lost]

        # Store the date the person will be diagnosed, as well as the date they took the test which will come back positive
        # Not for antigen tests?  date_diagnosed would interfere with later PCR.
        #ppl.date_diagnosed[true_positive_uids] = t + delay
        #ppl.date_pos_test[true_positive_uids] = t

        # False positivies
        if specificity < 1:
            non_infectious_uids = np.setdiff1d(inds, is_infectious_not_dx)
            false_positive_uids = cv.binomial_filter(1-specificity, non_infectious_uids)
        else:
            false_positive_uids = np.empty(0, dtype=np.int64)

        # At low prevalence, true_positive_uids will likely outnumber false_positive_uids
        return np.concatenate((true_positive_uids, false_positive_uids))


    def update(self, sim):
        '''
        Check for testing today and conduct tests if needed.
        True positives return via date_diagnosed, false positives are returned via this function.

        Fields include:
            * 'start_date': '2020-08-29',
            * 'repeat': None,
            * 'groups': ['students', 'teachers'],
            * 'coverage': 0.9,
            * 'sensitivity': 1,
            * 'delay': 1,
            * TODO: 'specificity': 1,
        '''

        # false_positive_uids = []
        ppl = sim.people
        t = sim.t
        ids_to_iso = {}
        for test in self.testing:
            if sim.t in test['t_vec']:
                undiagnosed_uids = cv.ifalsei(ppl.diagnosed, test['uids'])
                uids_to_test = cv.binomial_filter(test['coverage'], undiagnosed_uids)

                if self.school.verbose: print(sim.t, f'School {self.school.sid} of type {self.school.stype} is testing {len(uids_to_test)} today')

                if test['is_antigen']:
                    self.n_tested['Antigen'] += len(uids_to_test)
                    ag_pos_uids = self.antigen_test(uids_to_test, sim=sim, sym7d_sens=test['symp7d_sensitivity'], other_sens=test['other_sensitivity'], specificity=test['specificity'])

                    pcr_fu_uids = cv.binomial_filter(test['PCR_followup_perc'], ag_pos_uids)
                    ppl.test(pcr_fu_uids, test_sensitivity=1.0, test_delay=test['PCR_followup_delay'])
                    #sim.results['new_tests'][t] += len(pcr_fu_uids)
                    self.n_tested['PCR'] += len(pcr_fu_uids) # Also add follow-up PCR tests

                    ids_to_iso = {uid:t+test['PCR_followup_delay'] for uid in pcr_fu_uids}
                    non_pcr_uids = np.setdiff1d(ag_pos_uids, pcr_fu_uids)
                    ids_to_iso.update({uid:t+sim.pars['quar_period'] for uid in non_pcr_uids})
                else:
                    self.n_tested['PCR'] += len(uids_to_test)
                    ppl.test(uids_to_test, test_sensitivity=test['sensitivity'], test_delay=test['delay'])
                    #sim.results['new_tests'][t] += len(uids_to_test)
                    # N.B. No false positives for PCR

        return ids_to_iso



class SchoolStats(sc.prettyobj):
    ''' Reporter for tracking statistics associated with a school '''

    def __init__(self, school, sim):

        zero_vec = np.zeros(sim.npts)

        ppl = sim.people
        pop_scale = sim.pars['pop_scale']
        student_uids = cv.itruei(ppl.student_flag, school.uids)
        teacher_uids = cv.itruei(ppl.teacher_flag, school.uids)
        staff_uids = cv.itruei(ppl.staff_flag, school.uids)

        self.num_school_days = 0

        self.num = {
            'students':   len(student_uids) * pop_scale,
            'teachers':   len(teacher_uids) * pop_scale,
            'staff':      len(staff_uids) * pop_scale,
        }

        # Initialize results arrays
        base_result = {key:sc.dcp(zero_vec) for key in groups}
        self.infectious = sc.dcp(base_result)
        self.infectious_arrive_at_school = sc.dcp(base_result)
        self.infectious_stay_at_school = sc.dcp(base_result)
        self.newly_exposed = sc.dcp(base_result)
        self.scheduled = sc.dcp(base_result)
        self.in_person = sc.dcp(base_result)


    def update(self, sim, school):
        ''' Called on each day to update school statistics '''

        t = sim.t
        ppl = sim.people
        rescale = sim.rescale_vec[t]

        if school.ct_mgr.school_day:
            self.num_school_days += 1

        student_uids = cv.itruei(ppl.student_flag, school.uids)
        teacher_uids = cv.itruei(ppl.teacher_flag, school.uids)
        staff_uids = cv.itruei(ppl.staff_flag, school.uids)

        infectious_ids = {}
        for group, ids in zip(groups, [student_uids, teacher_uids, staff_uids]):
            infectious_ids[group] = cv.true(ppl.infectious[ids])
            self.infectious[group][t] = len(infectious_ids[group]) * rescale
            self.newly_exposed[group][t] = len(cv.true(ppl.date_exposed[ids] == t-1)) * rescale
            self.scheduled[group][t] = len(np.intersect1d(school.scheduled_uids, ids)) * rescale # Scheduled
            self.in_person[group][t] = len(np.intersect1d(school.uids_passed_screening, ids)) * rescale # Post-screening

        n_students_at_school = len(cv.itruei(ppl.student_flag * ppl.infectious, school.uids_arriving_at_school))
        n_teachers_at_school = len(cv.itruei(ppl.teacher_flag * ppl.infectious, school.uids_arriving_at_school))
        n_staff_at_school    = len(cv.itruei(ppl.staff_flag * ppl.infectious, school.uids_arriving_at_school))
        for group, count in zip(groups, [n_students_at_school, n_teachers_at_school, n_staff_at_school]):
            self.infectious_arrive_at_school[group][t] = count * rescale

        # Second "infectious_stay_at_school" effectively assumes "screen-positive" kids would be kept home from school in the first place
        n_students_passedscreening = len(cv.itruei(ppl.student_flag * ppl.infectious, school.uids_passed_screening))
        n_teachers_passedscreening = len(cv.itruei(ppl.teacher_flag * ppl.infectious, school.uids_passed_screening))
        n_staff_passedscreening    = len(cv.itruei(ppl.staff_flag * ppl.infectious, school.uids_passed_screening))
        for group, count in zip(groups, [n_students_passedscreening, n_teachers_passedscreening, n_staff_passedscreening]):
            self.infectious_stay_at_school[group][t] = count * rescale


    def get(self, school):
        ''' Called once on the final time step to return a dictionary that will be preserved in sim.school_info by school id. '''

        def sumdict(tsdict):
            ''' Sum over the time series for each group '''
            return {grp:np.sum(tsdict[grp]) for grp in groups}

        def firstday(tsdict):
            ''' Pull out the entry for the first day of school '''
            return {grp:tsdict[grp][school.start_day] for grp in groups}

        # Collect the output into a dict
        output = {
            'num': self.num,
            'infectious_first_day_school': firstday(self.infectious_stay_at_school),
            'infectious_stay_at_school': sumdict(self.infectious_stay_at_school),
            'newly_exposed': sumdict(self.newly_exposed),
            'scheduled': sumdict(self.scheduled),
            'in_person': sumdict(self.in_person),
            'num_school_days': self.num_school_days,
            'n_tested': school.testing.n_tested,
        }
        return output



class ContactManager(sc.prettyobj):
    ''' Base class for the contact managers below '''

    def __init__(self, uids, layer):
        self.uids = uids
        self.base_layer = layer
        self.school_day = False
        return

    def begin_day(self, date):
        ''' Called at the beginning of each day to configure the school layer -- implemented by each manager '''
        raise NotImplementedError()

    def find_contacts(self, uids):
        ''' Finds contacts of individuals listed in uids, including those who are absent from school -- implemented by each manager '''
        raise NotImplementedError()

    def remove_individuals(self, uids):
        ''' Remove one or more individual from the contact network '''
        rows = np.concatenate((
            np.isin(self.layer['p1'], uids).nonzero()[0],
            np.isin(self.layer['p2'], uids).nonzero()[0]))
        self.layer.pop_inds(rows)
        return

    def get_layer(self):
        ''' Return the layer '''
        return self.layer



class FullTimeContactManager(ContactManager):
    ''' Contact manager for regular 5-day-per-week school '''

    def __init__(self, sim, uids, layer):
        super().__init__(uids, layer)
        self.schedule = {
            'Monday':    'all',
            'Tuesday':   'all',
            'Wednesday': 'all',
            'Thursday':  'all',
            'Friday':    'all',
            'Saturday':  'weekend',
            'Sunday':    'weekend',
        }

    def begin_day(self, date):
        ''' Called at the beginning of each day to configure the school layer '''

        dayname = sc.readdate(date).strftime('%A')
        group = self.schedule[dayname]
        self.school_day = group == 'all'

        # Could modify layer based on group
        if group == 'all':
            # Start with the original layer, will remove uids at home later
            self.layer = sc.dcp(self.base_layer) # needed?
            uids = self.uids
        else:
            self.layer = cv.Layer() # Empty
            uids = np.empty(0, dtype='int64')
        return uids # Everyone is scheduled for school today, unless it's a weekend

    def find_contacts(self, uids):
        ''' Finds contacts of individuals listed in uids, including those who are absent from school '''
        return self.base_layer.find_contacts(uids)



class HybridContactManager(ContactManager):
    ''' Contact manager for hybrid school '''

    def __init__(self, sim, uids, layer):
        super().__init__(uids, layer)
        self.students = cv.itruei(sim.people.student_flag, self.uids)
        self.staff = cv.itruei(sim.people.staff_flag, self.uids)
        self.teachers = cv.itruei(sim.people.teacher_flag, self.uids)
        self.A_base_layer, self.B_base_layer = self.split_layer()
        self.schedule = {
            'Monday':    'A',
            'Tuesday':   'A',
            'Wednesday': 'distance',
            'Thursday':  'B',
            'Friday':    'B',
            'Saturday':  'weekend',
            'Sunday':    'weekend',
        }

    def split_layer(self):
        ''' Split the layer into A- and B-sublayers '''
        self.A_students = cv.binomial_filter(0.5, self.students)
        self.B_students = np.setdiff1d(self.students, self.A_students)

        self.A_group = np.concatenate((self.A_students, self.teachers, self.staff))
        self.B_group = np.concatenate((self.B_students, self.teachers, self.staff))

        A_layer = sc.dcp(self.base_layer)   # start with the full layer
        rows = np.concatenate((             # find all edges with a vertex in group B students
                np.isin(A_layer['p1'], self.B_students).nonzero()[0],
                np.isin(A_layer['p2'], self.B_students).nonzero()[0]))
        A_layer.pop_inds(rows)              # remove these edges from layer A

        B_layer = sc.dcp(self.base_layer)   # start with the full layer
        rows = np.concatenate((             # find all edges with a vertex in group A students
                np.isin(B_layer['p1'], self.A_students).nonzero()[0],
                np.isin(B_layer['p2'], self.A_students).nonzero()[0]))
        B_layer.pop_inds(rows)              # remove these edges from layer B

        return A_layer, B_layer

    def begin_day(self, date):
        ''' Called at the beginning of each day to configure the school layer '''
        dayname = sc.readdate(date).strftime('%A')
        group = self.schedule[dayname]
        self.school_day = group in ['A', 'B']

        # Could modify layer based on group
        if group == 'A':
            self.layer = sc.dcp(self.A_base_layer) # needed?
            uids = self.A_group
        elif group == 'B':
            self.layer = sc.dcp(self.B_base_layer) # needed?
            uids = self.B_group
        else:
            uids = np.empty(0, dtype='int64')
            self.layer = cv.Layer() # Empty

        return uids # Hybrid scheduling

    def find_contacts(self, uids):
        ''' Finds contacts of individuals listed in uids, including those who are absent from school.
            Look in A_base_layer as well as B_base_layer for uids and return contacts.
            Join together contacts of A-students in A-layer with B-students in B-layer
        '''
        contacts = np.concatenate((
            self.A_base_layer.find_contacts(uids),
            self.B_base_layer.find_contacts(uids) ))
        return contacts



class RemoteContactManager(ContactManager):
    ''' Contact manager for remote school '''

    def __init__(self, sim, uids, layer):
        super().__init__(uids, layer)
        self.base_layer = cv.Layer() # Empty base layer (ignore the passed-in layer)
        return

    def begin_day(self, date):
        ''' No students, so return an empty layer '''
        self.layer = cv.Layer() # Empty
        uids = np.empty(0, dtype='int64')
        return uids

    def remove_individuals(self, uids):
        ''' No individuals to remove, so just return '''
        return

    def find_contacts(self, uids):
        ''' No contacts because remote, return empty list '''
        return np.empty(0, dtype='int64')
