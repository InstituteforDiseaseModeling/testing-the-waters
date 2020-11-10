'''
Very simple test of population generation
'''

import numpy as np
import pylab as pl
import sciris as sc
import covasim as cv
import covasim_schools as cvsch

# Custom configuration
figsize = (24,20) # Customize to your screen resolution -- overwritten if maximize is true
dpi = 90 # Ditto, make smaller for smaller fonts etc
do_maximize = True # Fill the screen
to_json = True # Save results to JSON
outfile = 'school_pop_results.json' # Filename to save to, if JSON is saved

def test_school_pop(do_plot=False):
    ''' Test basic population creation '''

    pop = cvsch.make_population(pop_size=20e3, rand_seed=1, do_save=False)

    if do_plot:
        pop.plot()

    return pop


def plot_schools(pop):
    ''' Not a formal test, but a sanity check for school distributions '''
    keys = ['pk', 'es', 'ms', 'hs'] # Exclude universities for this analysis
    ppl_keys = ['all', 'students', 'teachers', 'staff']
    xpeople = np.arange(len(ppl_keys)) # X axis for people
    school_types_by_ind = {}
    for key,vals in pop.school_types.items():
        for val in vals:
            if key in keys:
                school_types_by_ind[val] = key

    results = {}
    for sc_id,sc_type in school_types_by_ind.items():
        thisres = sc.objdict()
        sc_inds = (pop.school_id == sc_id)
        thisres.all = cv.true(sc_inds)
        thisres.students = cv.true(np.array(pop.student_flag) * sc_inds)
        thisres.teachers = cv.true(np.array(pop.teacher_flag) * sc_inds)
        thisres.staff    = cv.true(np.array(pop.staff_flag) * sc_inds)
        results[sc_id] = thisres

    # Do plotting
    fig = pl.figure(figsize=figsize, dpi=dpi)
    pl.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.5, wspace=0.5)
    n_schools = len(results)
    n_cols = len(ppl_keys) + 1
    count = 0
    for sc_id in results.keys():
        count += 1
        school_type = school_types_by_ind[sc_id]
        ax = pl.subplot(n_schools, n_cols, count)
        thisres = results[sc_id]
        thisres.people_counts = [len(thisres[k]) for k in ppl_keys]
        ax.bar(xpeople, thisres.people_counts)
        ax.set_xticks(xpeople)
        ax.set_xticklabels(ppl_keys)
        title = f'School ID {sc_id}, school type {school_type}, total size: {len(thisres.all)}'
        ax.set_title(title)

        thisres.ages = sc.objdict()
        for key in ppl_keys:
            count += 1
            ax = pl.subplot(n_schools, n_cols, count)
            thisres.ages[key] = pop.age[thisres[key]]
            pl.hist(thisres.ages[key])
            ax.set_title(f'Ages for {key} in school {sc_id} ({school_type})')

    if do_maximize:
        cv.maximize(fig=fig)

    if to_json:
        sc.savejson(outfile, results, indent=2)

    return results


if __name__ == '__main__':
    pop = test_school_pop(do_plot=False)
    results = plot_schools(pop)

