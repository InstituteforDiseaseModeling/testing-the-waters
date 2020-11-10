'''
Pre-generate the synthpops population including school types. Takes ~10s per seed
if running with 20,000 people; or roughly 10 times longer for 225,000 people.

After running this script, you will need to manually move the files generated in 
the inputs folder into the main results folder (v20201009).

Warning: this script is quite memory intensive. If you do not have much RAM, you may
wish to turn off parallelization.
'''

import psutil
import sciris  as sc
import covasim_schools as cvsch


# This must be in a main block for parallelization to work on Windows
if __name__ == '__main__':

    pop_size = 20e3
    seeds = [0,1,2,3,4]
    parallelize = True

    if parallelize:
        ram = psutil.virtual_memory().available/1e9
        required = 1*len(seeds)*pop_size/225e3 # 8 GB per 225e3 people
        if required < ram:
            print(f'You have {ram} GB of RAM, and this is estimated to require {required} GB: you should be fine')
        else:
            print(f'You have {ram:0.2f} GB of RAM, but this is estimated to require {required} GB -- you may wish to terminate this process or else your computer could run out of memory')
        sc.parallelize(cvsch.make_population, kwargs={'pop_size':pop_size}, iterkwargs={'rand_seed':seeds}) # Run them in parallel
    else:
        for seed in seeds:
            cvsch.make_population(pop_size=pop_size, rand_seed=seed)
