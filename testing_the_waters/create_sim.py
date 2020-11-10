'''
Main script to create a simulation, the base on which analysis is conducted.
Used in calibration and other downstream activities.
'''

import os
import covasim as cv
import sciris as sc
import covasim_schools as cvsch


def define_pars(which='best', kind='default', ):
    ''' Define the parameter best guesses and bounds -- used for calibration '''

    pardata = {}
    if kind in ['default', 'both']:
        # The following three parameters are adjusted during calibration.
        pardata.update(dict(
            pop_infected = [200, 100, 300],
            change_beta=[0.6, 0.45, 0.75],
            symp_prob=[0.3, 0.1, 0.5],
        ))

    output = {}
    for key,arr in pardata.items():
        if which == 'best':
            output[key] = arr[0]
        elif which == 'bounds':
            output[key] = arr[1:3]

    return output


def create_sim(params=None, pop_size=2.25e5, rand_seed=1, folder=None, popfile_stem=None,
               children_equally_sus=False, alternate_symptomaticity=False, max_pop_seeds=5, load_pop=True, save_pop=False, people=None,
               label=None, verbose=0, **kwargs):
    '''
    Create the simulation for use with schools. This is the main function used to
    create the sim object.

    Args:
        params (dict): the parameters to use for the simulation
        pop_size (int): the number of people (merged into parameters)
        rand_seed (int): the random seed to use (merged into parameters)
        folder (str): where to look for the population file
        popfile_stem (str): filename of population file, minus random seed (which gets added)
        children_equally_sus (bool): whether children should be equally susceptible as adults (for sensitivity)
        alternate_symptomaticity (bool): whether to use symptoms by age from Table 1 in https://arxiv.org/pdf/2006.08471.pdf
        max_pop_seeds (int): maximum number of populations to generate (for use with different random seeds)
        load_pop (bool): whether to load people from disk (otherwise, use supplied or create afresh)
        save_pop (bool): if a population is being generated, whether to save
        people (People): if supplied, use instead of loading from file
        label (str): a name for the simulation
        verbose (float): level of verbosity to use (merged into parameters)
        kwargs (dict): merged with params

    Returns:
        A sim instance
    '''

    # Handle parameters and merge together different sets of defaults

    default_pars = dict(
        pop_size       = pop_size,
        pop_scale      = 1,
        pop_type       = 'synthpops',
        rescale        = False, # True causes problems
        verbose        = verbose,
        start_day      = '2020-09-01',
        end_day        = '2021-01-31',
        rand_seed      = rand_seed
    )

    p = sc.objdict(sc.mergedicts(default_pars, define_pars(which='best', kind='both'), params, kwargs)) # Get default parameter values


    #%% Define interventions
    symp_prob = p.pop('symp_prob')
    change_beta = p.pop('change_beta')

    tp_pars = dict(
        symp_prob = symp_prob,
        asymp_prob = 0.0022,
        symp_quar_prob = symp_prob,
        asymp_quar_prob = 0.001,
        test_delay = 2.0,
    )

    ct_pars = dict(
        trace_probs = {'w': 0.1, 'c': 0, 'h': 0.9, 's': 0.8}, # N.B. 's' will be ignored if using the Schools class
        trace_time  = {'w': 2,   'c': 0, 'h': 1,   's': 2},
    )

    cb_pars = dict(
        changes=change_beta,
        layers=['w', 'c'],
        label='NPI_work_community',
    )

    ce_pars = dict(
        changes=0.65,
        layers=['w', 'c'],
        label='close_work_community'
    )

    # Define Covasim interventions
    interventions = [
        cv.test_prob(start_day=p.start_day, **tp_pars),
        cv.contact_tracing(start_day=p.start_day, **ct_pars),
        cv.change_beta(days=p.start_day, **cb_pars),
        cv.clip_edges(days=p.start_day, **ce_pars),
        # N.B. Schools are not closed in create_sim, must be handled outside this function
    ]
    for interv in interventions:
        interv.do_plot = False


    #%% Handle population -- NB, although called popfile, might be a People object
    if load_pop: # Load from disk -- normal usage
        if popfile_stem is None:
            popfile_stem = os.path.join('inputs', f'kc_synthpops_clustered_{int(pop_size)}_withstaff_seed')
        if folder is not None:
            popfile_stem = os.path.join(folder, popfile_stem) # Prepend user folder
        pop_seed = p.rand_seed % max_pop_seeds
        popfile = popfile_stem + str(pop_seed) + '.ppl'
        print(f'Note: loading population from {popfile}')
    elif people is not None: # People is supplied; use that
        popfile = people
        print('Note: using supplied people')
    else: # Generate
        print('Note: population not supplied; regenerating...')
        popfile = cvsch.make_population(pop_size=p.pop_size, rand_seed=p.rand_seed, max_pop_seeds=max_pop_seeds, do_save=False)

    # Create sim
    sim = cv.Sim(p, popfile=popfile, load_pop=True, label=label, interventions=interventions)

    # Modify sim for variations
    # 1. Children equally as susceptible as adults 20-64
    if children_equally_sus:
        prog = sim.pars['prognoses']
        ages = prog['age_cutoffs']
        sus_ORs = prog['sus_ORs']
        sus_ORs[ages<=20] = 1
        prog['sus_ORs'] = sus_ORs

    # 2. Lower levels of symptomaticity in children
    if alternate_symptomaticity:
        prog = sim.pars['prognoses']
        ages = prog['age_cutoffs']
        symp_probs = prog['symp_probs']
        # Source: table 1 from https://arxiv.org/pdf/2006.08471.pdf
        symp_probs[:] = 0.6456
        symp_probs[ages<80] = 0.3546
        symp_probs[ages<60] = 0.3054
        symp_probs[ages<40] = 0.2241
        symp_probs[ages<20] = 0.1809
        prog['symp_probs'] = symp_probs

    return sim
