'''
Plot countermeasure scenarios, resulting in Figure 6.  Call this after running countermeasure_scenarios.py.
'''

import os
import covasim as cv
import sciris as sc
import covasim.misc as cvm
import numpy as np
import pandas as pd
import matplotlib as mplt
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from calibrate_model import evaluate_sim
from pathlib import Path

# Global plotting styles
font_size = 18
font_style = 'Roboto Condensed'
mplt.rcParams['font.size'] = font_size
mplt.rcParams['font.family'] = font_style

pop_size = 2.25e5

folder = 'v20201019'
variant = 'countermeasures_v2_0-30'
school_scenario = 'with_countermeasures'
cachefn = os.path.join(folder, 'msims', f'{variant}.sims')

imgdir = os.path.join(folder, 'img_'+variant)
Path(imgdir).mkdir(parents=True, exist_ok=True)

print(f'loading {cachefn}')
#sims = cv.MultiSim.load(cachefn).sims  # If loading a *.msim object
sims = cv.load(cachefn)                 # If loading a *.sims object

results = []
groups = ['students', 'teachers', 'staff']

scen_names = sc.odict({ # key1
    'as_normal': 'As Normal',
    'with_countermeasures': 'Normal with\nCountermeasures',
    'all_hybrid': 'Hybrid with\nCountermeasures',
    'k5': 'K-5 In-Person\nOthers Remote',
    'all_remote': 'All Remote',
})
scen_order = scen_names.values()

test_names = sc.odict({ # key2
    'None':                                  ('No diagnostic screening',                        'gray'),
    'PCR 1w prior':                          ('PCR\n1w prior\n1d delay',                        (0.8584083044982699, 0.9134486735870818, 0.9645674740484429, 1.0)),
    'Antigen every 1w teach&staff, PCR f/u': ('Antigen\nEvery 1w\nTeachers & Staff\nPCR f/u',   (0.9882352941176471, 0.732072279892349, 0.6299269511726259, 1.0)),
    'PCR every 2w':                          ('PCR\nFortnightly\n1d delay',                     (0.5356862745098039, 0.746082276047674, 0.8642522106881968, 1.0)),
    'Antigen every 2w, no f/u':              ('Antigen\nFortnightly\nno f/u',                   (0.7925720876585928, 0.09328719723183392, 0.11298731257208766, 1.0)),
    'Antigen every 2w, PCR f/u':             ('Fortnightly antigen with PCR f/u',               (0.9835755478662053, 0.4127950788158401, 0.28835063437139563, 1.0)),
    'PCR every 1w':                          ('PCR\nWeekly\n1d delay',                          (0.32628988850442137, 0.6186236063052672, 0.802798923490965, 1.0)),
    'PCR every 1d':                          ('PCR\nDaily\nNo delay',                           (0.16696655132641292, 0.48069204152249134, 0.7291503267973857, 1.0)),
})
test_order = [v[0] for k,v in test_names.items() if k in ['None', 'Antigen every 2w, PCR f/u']]
test_hue = {v[0]:v[1] for v in test_names.values()}

sens_names = sc.odict({ # key3
    'NPI_screening_tracing': 'No countermeasures', #'No NPI, screening, or tracing',
    'NPI_screening': 'With tracing', #'No NPI or screening',
    'screening_tracing': 'With NPI', #'No screening or tracing',
    'screening': 'With NPI and tracing', #'No daily screening',
    'NPI_tracing': 'With screening', #'No NPI or tracing',
    'NPI': 'With screening and tracing', #'No NPI reduction',
    'tracing': 'With NPI and screening', #'No contact tracing',
    'baseline': 'With NPI, screening, and tracing', #'Baseline',
})
sens_order = sens_names.values()


# MAIN LOOP OVER SIMS.  Collect relevant stats on each.
for sim in sims:
    sim.key2 = test_names[sim.key2][0] if sim.key2 in test_names else sim.key2
    sim.key3 = sens_names[sim.key3] if sim.key3 in sens_names else sim.key3

    first_date = '2020-11-02'
    first_school_day = sim.day(first_date)
    last_date = '2021-01-31'
    last_school_day = sim.day(last_date)
    ret = {
        'key1': sim.key1,
        'key2': sim.key2,
        'key3': sim.key3,
    }

    perf = evaluate_sim(sim)
    ret.update(perf)

    n_schools = {'es':0, 'ms':0, 'hs':0}
    n_schools_with_inf_d1 = {'es':0, 'ms':0, 'hs':0}

    grp_dict = {'Students': ['students'], 'Teachers & Staff': ['teachers', 'staff'], 'Students, Teachers, and Staff': ['students', 'teachers', 'staff']}
    perc_inperson_days_lost = {k:[] for k in grp_dict.keys()}
    attackrate = {k:[] for k in grp_dict.keys()}
    count = {k:0 for k in grp_dict.keys()}
    exposed = {k:0 for k in grp_dict.keys()}
    inperson_days = {k:0 for k in grp_dict.keys()}
    possible_days = {k:0 for k in grp_dict.keys()}

    for sid,stats in sim.school_stats.items():
        if stats['type'] not in ['es', 'ms', 'hs']:
            continue

        inf = stats['infectious']
        inf_at_sch = stats['infectious_stay_at_school'] # Post-screening
        in_person = stats['in_person']
        exp = stats['newly_exposed']
        num_school_days = stats['num_school_days']
        possible_school_days = np.busday_count(first_date, last_date)
        n_exp = {}
        for grp in groups:
            n_exp[grp] = np.sum(exp[grp])

        for gkey, grps in grp_dict.items():
            in_person_days = scheduled_person_days = num_exposed = num_people = 0
            for grp in grps:
                in_person_days += np.sum(in_person[grp])
                scheduled_person_days += num_school_days * stats['num'][grp]
                num_exposed += n_exp[grp]
                num_people += stats['num'][grp]
            perc_inperson_days_lost[gkey].append(
                100*(scheduled_person_days - in_person_days)/scheduled_person_days if scheduled_person_days > 0 else 100
            )
            attackrate[gkey].append( 100 * num_exposed / num_people)

            exposed[gkey] += n_exp[grp]
            count[gkey] += stats['num'][grp]
            inperson_days[gkey] += in_person_days
            possible_days[gkey] += possible_school_days*num_people

        n_schools[stats['type']] += 1
        if inf_at_sch['students'][first_school_day] + inf_at_sch['teachers'][first_school_day] + inf_at_sch['staff'][first_school_day] > 0:
            n_schools_with_inf_d1[stats['type']] += 1

    for stype in ['es', 'ms', 'hs']:
        ret[f'{stype}_perc_d1'] = 100 * n_schools_with_inf_d1[stype] / n_schools[stype]

    # Deciding between district and school perspective here
    for gkey in grp_dict.keys():
        ret[f'perc_inperson_days_lost_{gkey}'] = 100*(possible_days[gkey]-inperson_days[gkey])/possible_days[gkey] #np.mean(perc_inperson_days_lost[gkey])
        ret[f'attackrate_{gkey}'] = 100*exposed[gkey] / count[gkey] #np.mean(attackrate[gkey])
        ret[f'count_{gkey}'] = np.sum(count[gkey])

    results.append(ret)

# PRODUCE PLOTS

df = pd.DataFrame(results)

# Frac in-person days lost
d = pd.melt(df, id_vars=['key1', 'key2', 'key3'], value_vars=[f'perc_inperson_days_lost_{gkey}' for gkey in grp_dict.keys()], var_name='Group', value_name='Days lost (%)')
d.replace( {'Group': {f'perc_inperson_days_lost_{gkey}':gkey for gkey in grp_dict.keys()}}, inplace=True)
d = d.loc[d['key1']==school_scenario] # K-5 only
g = sns.FacetGrid(data=d, row='Group', height=4, aspect=3, row_order=['Teachers & Staff', 'Students'], legend_out=False)
g.map_dataframe( sns.barplot, x='key2', y='Days lost (%)', hue='key3', order=test_order, hue_order=sens_order, palette='Set2')
g.set_titles(row_template="{row_name}", fontsize=24)
g.set_axis_labels(y_var="Days lost (%)")
plt.tight_layout()

for axi, ax in enumerate(g.axes.flat):
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + (axi+1)*box.height * 0.1, box.width, box.height * 0.9])
g.axes.flat[1].legend(loc='upper center',bbox_to_anchor=(0.48,-0.16), ncol=4, fontsize=14)

cv.savefig(os.path.join(imgdir, '3mInPersonDaysLost_countermeasures.png'), dpi=300)


# Attack rate
d = pd.melt(df, id_vars=['key1', 'key2', 'key3'], value_vars=[f'attackrate_{gkey}' for gkey in grp_dict.keys()], var_name='Group', value_name='Cum Inc (%)')
d.replace( {'Group': {f'attackrate_{gkey}':gkey for gkey in grp_dict.keys()}}, inplace=True)
d = d.loc[d['key1']==school_scenario] # K-5 only
d = d.loc[d['Group']!='Students, Teachers, and Staff']
for aspect, fontsize in zip([2.5, 3], [12,14]):
    g = sns.FacetGrid(data=d, row='Group', height=4, aspect=aspect, row_order=['Teachers & Staff', 'Students'], legend_out=False) # col='key1', 
    g.map_dataframe( sns.barplot, x='key2', y='Cum Inc (%)', hue='key3', order=test_order, hue_order=sens_order, palette='Set2')
    g.set_titles(row_template="{row_name}")
    g.set_axis_labels(y_var="3-Month Attack Rate (%)")
    plt.tight_layout()

    for axi, ax in enumerate(g.axes.flat):
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + (axi+1)*box.height * 0.1, box.width, box.height * 0.9])
    g.axes.flat[1].legend(loc='upper center',bbox_to_anchor=(0.48,-0.16), ncol=4, fontsize=fontsize)

    cv.savefig(os.path.join(imgdir, f'3mAttackRate_countermeasures_{aspect}.png'), dpi=300)

# Attack rate (simple)
d = pd.melt(df, id_vars=['key1', 'key2', 'key3'], value_vars=[f'attackrate_{gkey}' for gkey in grp_dict.keys()], var_name='Group', value_name='Cum Inc (%)')
d.replace( {'Group': {f'attackrate_{gkey}':gkey for gkey in grp_dict.keys()}}, inplace=True)
d = d.loc[d['key1']==school_scenario] # K-5 only
d = d.loc[d['Group']=='Students, Teachers, and Staff']
for aspect, fontsize in zip([1.3], [12,14]):
    g = sns.FacetGrid(data=d, row='Group', height=8, aspect=aspect, row_order=['Students, Teachers, and Staff'], legend_out=False) # col='key1', 
    g.map_dataframe( sns.barplot, x='key2', y='Cum Inc (%)', hue='key3', order=test_order, hue_order=sens_order, palette='Set2')
    g.set_titles(row_template="{row_name}")
    g.set_axis_labels(y_var="Percent acquiring COVID-19 over 3 months")
    g.add_legend()
    plt.tight_layout()

    cv.savefig(os.path.join(imgdir, f'3mAttackRate_countermeasures_simple_{aspect}.png'), dpi=300)

# Attack rate (very simple variant)
d = pd.melt(df, id_vars=['key1', 'key2', 'key3'], value_vars=[f'attackrate_{gkey}' for gkey in grp_dict.keys()], var_name='Group', value_name='Cum Inc (%)')
d.replace( {'Group': {f'attackrate_{gkey}':gkey for gkey in grp_dict.keys()}}, inplace=True)
d = d.loc[d['key1']==school_scenario] # K-5 only
d = d.loc[d['Group']=='Students, Teachers, and Staff']
test_order = [v[0] for k,v in test_names.items() if k in ['None']]

num_countermeasures = { # key3
    'NPI_screening_tracing': 'None',
    'NPI_screening': 'One',
    'screening_tracing': 'One',
    'screening': 'Two',
    'NPI_tracing': 'One',
    'NPI': 'Two',
    'tracing': 'Two',
    'baseline': 'All Three',
}

d['Num Countermeasures'] = d['key3']
d.replace( {'Num Countermeasures': num_countermeasures}, inplace=True)
for figsize in [(12,8), (12,9.5)]:
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(data=d, x='Num Countermeasures', y='Cum Inc (%)', hue='key3', order=['None', 'One', 'Two', 'All Three'], hue_order=sens_order, palette='Set2', ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('Percent acquiring COVID-19 over 3 months')
    ax.legend()
    plt.tight_layout()

    cv.savefig(os.path.join(imgdir, f'3mAttackRate_countermeasures_verysimple_{aspect}.png'), dpi=300)
