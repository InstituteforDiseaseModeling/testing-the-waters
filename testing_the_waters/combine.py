'''
Example script to combine several *.msim or *.sims objects into a single sims object.
'''

import covasim as cv
import os

folder = 'v20201019'
base = 'sensitivity_v3_0-10'

fns = [
    f'{base}_baseline.msim',
    f'{base}_lower_sens_spec.msim',
    f'{base}_no_NPI_reduction.msim',
    f'{base}_lower_random_screening.msim',
    f'{base}_no_screening.msim',
    f'{base}_lower_coverage.msim',
    f'{base}_alt_symp.msim',
    f'{base}_children_equally_sus.msim',
    f'{base}_increased_mobility.msim',
    f'{base}_broken_bubbles.msim',
]
fns = [os.path.join(folder, 'msims', fn) for fn in fns]

sims = []

# Loop to load in each file and combine results into the sims list
for fn in fns:
    print(f'Loading {fn}')
    ext = os.path.splitext(fn)[1]
    if ext == '.sims':
        sims += cv.load(fn)
    elif ext == '.msim':
        msim = cv.MultiSim.load(fn)
        sims += msim.sims
    else:
        print('ERROR')

fn = os.path.join(folder, 'msims', f'{base}.sims')
print(f'Saving to {fn}')
cv.save(fn, sims)
