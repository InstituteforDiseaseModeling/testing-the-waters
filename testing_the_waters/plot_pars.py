'''
Simple script to read in one of the pars_* files and create a few histograms.
'''

import os
import sciris as sc
import matplotlib.pyplot as plt
import seaborn as sns
import create_sim as cs

folder = 'v20201019'
calibfile = os.path.join(folder, 'pars_cases_begin=75_cases_end=75_re=1.0_prevalence=0.002_yield=0.024_tests=225_pop_size=225000_alternate_symptomaticity.json')
par_inds = (0,30) # Read in the first 30, as this is what's typically used in analysis

par_list = sc.loadjson(calibfile)[par_inds[0]:par_inds[1]]
keys = par_list[0]['pars'].keys()

ret = {}
for k in keys:
    ret[k] = [p['pars'][k] for p in par_list]

lims = cs.define_pars(which='bounds')
f, axv = plt.subplots(1, len(keys), figsize=(12,8))
for k,ax in zip(keys, axv):
    sns.distplot(ret[k], rug=True, ax=ax)
    lim = lims[k]
    ax.axvline(x=lim[0], c='red')
    ax.axvline(x=lim[1], c='red')
    ax.set_xlabel(k)

plt.show()
