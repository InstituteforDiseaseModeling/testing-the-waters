'''
Very simple test of sim creation -- not part of the main library
'''

import covasim as cv
from testing_the_waters import create_sim as cs

def test_sims(do_plot=False):
    s1 = cs.create_sim(pop_size=10e3, load_pop=False, label='Default')
    s2 = cs.create_sim(pop_size=10e3, load_pop=False, rand_seed=23948, label='Different seed')
    s3 = cs.create_sim(pop_size=10e3, load_pop=False, beta=0.03, label='High beta')

    msim = cv.MultiSim([s1, s2, s3])
    msim.run()

    if do_plot:
        msim.plot()

    return msim

if __name__ == '__main__':
    msim = test_sims(do_plot=True)
