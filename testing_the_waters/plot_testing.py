'''
Plot diagnostic screening scenarios, resulting in Figures 1, 2, 4, and 7.
Call this after running run_testing_scenarios.py.
'''

import os
from pathlib import Path
import covasim as cv
import sciris as sc
import numpy as np
import pandas as pd
import matplotlib as mplt
import matplotlib.pyplot as plt
import seaborn as sns
from calibrate_model import evaluate_sim # Local import


# Which figures to plot
to_plot = {'Fig. 1': True, # 3-month attack rate for students and teachers + staff
           'Combined attack rate': False, # Like Fig. 1, but with students, staff, and teachers combined
           'Separate attack rates': False, # Like Fig. 1, but plot attack rates in separate figures
           'Fig. 2': True, # Population-wide reproduction number
           'Fig. 4': True, # Proportion of days that are remote instead of in-person learning
           'Additional tests': False, # Additional diagnostic tests required
           }

# Global plotting styles
font_size = 18
font_style = 'Roboto Condensed'
mplt.rcParams['font.size'] = font_size
mplt.rcParams['font.family'] = font_style

# Other configuration
pop_size = 2.25e5
folder = 'v20201019'
variant = 'final_20201026_0-30'

cachefn = os.path.join(folder, 'sims', f'{variant}.sims') # Might need to change the extension here, depending in combine.py was used
simple = False # Boolean flag to select a subset of the scenarios

variant = variant + '_simple' if simple else variant
imgdir = os.path.join(folder, 'img_'+variant)
Path(imgdir).mkdir(parents=True, exist_ok=True)

for_presentation = False # Choose between report style and presentation style (different aspect ratio)
figsize = (12,8) if for_presentation else (12,9.5)
aspect = 3 if for_presentation else 2.5

T = sc.tic() # Start timing

print(f'Loading {cachefn}')
sims = cv.load(cachefn) # Use for *.sims

def ax_col(ax, col='blue'):
    ''' Set standard axis configurations '''
    ax.spines['left'].set_color(col)
    ax.tick_params(axis='y', colors=col)
    ax.xaxis.label.set_color(col)

results = []
byschool = []
groups = ['students', 'teachers', 'staff']

scen_names = sc.odict({ # key1
    'as_normal': 'Full Schedule\nNo Countermeasures',
    'with_countermeasures': 'Full Schedule',
    'all_hybrid': 'Hybrid',
    'k5': 'K-5 In-Person\nOthers Remote',
    'all_remote': 'All Remote',
})
scen_order = scen_names.keys()

blues = plt.cm.get_cmap('Blues')
reds = plt.cm.get_cmap('Reds')
test_names = sc.odict({ # key2
    'None':                                     ('No diagnostic screening',                         'gray'),
    'PCR 1w prior':                             ('PCR one week prior, 1d delay',                    blues(1/5)),
    'Antigen every 1w teach&staff, PCR f/u':    ('Weekly antigen for teachers & staff, PCR f/u',    reds(1/5)),
    'Antigen every 2w, no f/u':                 ('Fortnightly antigen, no f/u',                     reds(2/5)),
    'Antigen every 2w, PCR f/u':                ('Fortnightly antigen, PCR f/u',                    reds(3/5)),
    'PCR every 2w':                             ('Fortnightly PCR, 1d delay',                       blues(2/5)),
    'Antigen every 1w, PCR f/u':                ('Weekly antigen, PCR f/u',                         reds(4/5)),
    'PCR every 1w':                             ('Weekly PCR, 1d delay',                            blues(3/5)),
    'PCR every 1d':                             ('Daily PCR, no delay',                             blues(4/5)),
})

if simple:
    # Select a subset
    test_order = [v[0] for k,v in test_names.items() if k in [ 'None', 'PCR 1w prior', 'Antigen every 1w teach&staff, PCR f/u', 'Antigen every 2w, PCR f/u', 'PCR every 2w', 'Antigen every 1w, PCR f/u'] ]
else:
    test_order = [v[0] for k,v in test_names.items()]
test_hue = {v[0]:v[1] for v in test_names.values()}



#%% Process the simulations

for sim in sims:
    first_school_day = sim.day('2020-11-02')
    last_school_day = sim.day('2021-01-31')
    sim.key2 = test_names[sim.key2][0] if sim.key2 in test_names else sim.key2

    ret = {
        'key1': sim.key1,
        'key2': sim.key2,
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

    first_date = '2020-11-02'
    first = sim.day(first_date)
    last_date = '2021-01-31'
    last = sim.day(last_date)

    for sid,stats in sim.school_stats.items():
        if stats['type'] not in ['es', 'ms', 'hs']:
            continue

        inf_at_sch = stats['infectious_stay_at_school'] # Post-screening
        inf_first = stats['infectious_first_day_school'] # Post-screening
        in_person = stats['in_person']
        exp = stats['newly_exposed']
        num_school_days = stats['num_school_days']
        possible_school_days = np.busday_count(first_date, last_date)
        n_exp = {}
        for grp in groups:
            n_exp[grp] = exp[grp]

        for gkey, grps in grp_dict.items():
            in_person_days = scheduled_person_days = num_exposed = num_people = 0
            for grp in grps:
                in_person_days += in_person[grp]
                scheduled_person_days += num_school_days * stats['num'][grp]
                num_exposed += n_exp[grp]
                num_people += stats['num'][grp]
                exposed[gkey] += n_exp[grp]
                count[gkey] += stats['num'][grp]

            perc_inperson_days_lost[gkey].append(
                100*(scheduled_person_days - in_person_days)/scheduled_person_days if scheduled_person_days > 0 else 100
            )
            attackrate[gkey].append( 100 * num_exposed / num_people)

            inperson_days[gkey] += in_person_days
            possible_days[gkey] += possible_school_days*num_people

        n_schools[stats['type']] += 1
        if sum([inf_first[g] for g in groups]) > 0:
            n_schools_with_inf_d1[stats['type']] += 1

        byschool.append({
            'sid': sid,
            'type': stats['type'],
            'key1': sim.key1, # Filtered to just one scenario (key1)
            'key2': sim.key2,
            'n_students': stats['num']['students'],
            'n': sum([stats['num'][g] for g in groups]),
            'd1 infectious': sum([inf_first[g] for g in groups]),
            'd1 bool': sum([inf_first[g] for g in groups]) > 0,
            'PCR': stats['n_tested']['PCR'],
            'Antigen': stats['n_tested']['Antigen'],
            'Days': last_school_day - first_school_day,
            'Pop*Scale': sim.pars['pop_size']*sim.pars['pop_scale'],
        })

    for stype in ['es', 'ms', 'hs']:
        ret[f'{stype}_perc_d1'] = 100 * n_schools_with_inf_d1[stype] / n_schools[stype]

    # Deciding between district and school perspective here
    for gkey in grp_dict.keys():
        ret[f'perc_inperson_days_lost_{gkey}'] = 100*(possible_days[gkey]-inperson_days[gkey])/possible_days[gkey] #np.mean(perc_inperson_days_lost[gkey])
        ret[f'attackrate_{gkey}'] = 100*exposed[gkey] / count[gkey] #np.mean(attackrate[gkey])
        ret[f'count_{gkey}'] = np.sum(count[gkey])

    results.append(ret)

# Convert results to a dataframe
df = pd.DataFrame(results)


################
#%% PLOT RESULTS
################


#%% Fig. 1

# Attack rate - two axes in one plot
d = pd.melt(df, id_vars=['key1', 'key2'], value_vars=[f'attackrate_{gkey}' for gkey in grp_dict.keys()], var_name='Group', value_name='Cum Inc (%)')
d.replace( {'Group': {f'attackrate_{gkey}':gkey for gkey in grp_dict.keys()}}, inplace=True)
d.replace( {'key1': scen_names}, inplace=True)
d.to_csv(os.path.join(imgdir, f'3mAttackRate_combined.csv'))
so = [scen_names[x] for x in scen_order]

if to_plot['Fig. 1']:
    fig = plt.figure('Fig. 1: 3-month attack rate', constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(20, 50)

    ts = d.loc[(d['Group']=='Teachers & Staff')]

    # Teachers and Staff
    ax = fig.add_subplot(gs[0, :])
    ax.axis('off')
    ax.text(0.5, 0, 'Teachers & Staff', horizontalalignment='center')

    # Top left
    ax = fig.add_subplot(gs[1:8, 0:10])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    sns.barplot(data=ts, x='key1', y='Cum Inc (%)', hue='key2', hue_order=test_order, order=so, palette=test_hue)
    ax.get_legend().remove()
    ax.set_xlim([-0.5,0.5])
    ax.set_ylim([0, 60])
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('3-Month Attack Rate (%)')

    # Top right
    ax = fig.add_subplot(gs[1:8, 10:])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    sns.barplot(data=ts, x='key1', y='Cum Inc (%)', hue='key2', hue_order=test_order, order=so, palette=test_hue)
    ax.get_legend().remove()
    ax.set_xlim([0.5,4.5])
    ax.set_ylim([0, 6])
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    if simple:
        ax_col(ax)

    stu = d.loc[(d['Group']=='Students')]

    # Students
    ax = fig.add_subplot(gs[8, :])
    ax.axis('off')
    ax.text(0.5, 0, 'Students', horizontalalignment='center')

    # Bottom left
    ax = fig.add_subplot(gs[9:16, 0:10])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    sns.barplot(data=stu, x='key1', y='Cum Inc (%)', hue='key2', hue_order=test_order, order=so, palette=test_hue)
    ax.get_legend().remove()
    ax.set_xlim([-0.5,0.5])
    ax.set_ylim([0, 60])
    ax.set_xlabel('')
    ax.set_ylabel('3-Month Attack Rate (%)')

    # Bottom right
    ax = fig.add_subplot(gs[9:16, 10:])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    sns.barplot(data=stu, x='key1', y='Cum Inc (%)', hue='key2', hue_order=test_order, order=so, palette=test_hue)
    hnd, lbl = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    ax.set_xlim([0.5,4.5])
    ax.set_ylim([0, 6])
    ax.set_ylabel('')
    ax.set_xlabel('')
    if simple:
        ax_col(ax)

    ax = fig.add_subplot(gs[16:, :])
    ax.axis('off')
    ax.legend(hnd, lbl, ncol=3, loc='center', fontsize=16)

    cv.savefig(os.path.join(imgdir, f'3mAttackRate_combined_{figsize}.png'), dpi=300)



#%% Combined attack rate

if to_plot['Combined attack rate']:

    # Attack rate - All school community members
    d = pd.melt(df, id_vars=['key1', 'key2'], value_vars=[f'attackrate_{gkey}' for gkey in grp_dict.keys()], var_name='Group', value_name='Cum Inc (%)')
    d.replace( {'Group': {f'attackrate_{gkey}':gkey for gkey in grp_dict.keys()}}, inplace=True)
    d.replace( {'key1': scen_names}, inplace=True)
    d.to_csv(os.path.join(imgdir, f'3mAttackRate_sts.csv'))
    so = [scen_names[x] for x in scen_order]

    fig = plt.figure('Combined attack rate', constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(20, 50)

    # Teachers and Staff
    ax = fig.add_subplot(gs[0, :])
    ax.axis('off')
    ax.text(0.5, 0, 'Students, Teachers, and Staff', horizontalalignment='center')

    sts = d.loc[(d['Group']=='Students, Teachers, and Staff')]

    # Left
    ax = fig.add_subplot(gs[1:18, 0:16])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    sns.barplot(data=sts, x='key1', y='Cum Inc (%)', hue='key2', hue_order=test_order, order=so, palette=test_hue)
    ax.get_legend().remove()
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([0, 45])
    ax.set_yticks([0,10,20,30,40])
    ax.set_xlabel('')
    ax.set_ylabel('Percent acquiring COVID-19 over 3 months')

    # Right
    ax = fig.add_subplot(gs[1:18, 16:])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(length=0)
    sns.barplot(data=sts, x='key1', y='Cum Inc (%)', hue='key2', hue_order=test_order, order=so, palette=test_hue)
    hnd, lbl = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    ax.set_xlim([0.5,4.5])
    ax.set_ylim([0, 4.5])
    ax.set_yticks([0,1,2,3,4])
    ax.set_ylabel('')
    ax.set_xlabel('')
    if simple:
        ax_col(ax)

    ax = fig.add_subplot(gs[18:, :])
    ax.axis('off')
    ax.legend(hnd, lbl, ncol=3, loc='center', fontsize=15)

    cv.savefig(os.path.join(imgdir, f'3mAttackRate_sts_{figsize}.png'), dpi=300)



#%% Separate attack rates

if to_plot['Separate attack rates']:

    # Attack rate
    for name in ['all', 'no_normal']:
        d = pd.melt(df, id_vars=['key1', 'key2'], value_vars=[f'attackrate_{gkey}' for gkey in grp_dict.keys()], var_name='Group', value_name='Cum Inc (%)')
        d.replace( {'Group': {f'attackrate_{gkey}':gkey for gkey in grp_dict.keys()}}, inplace=True)
        d.to_csv(os.path.join(imgdir, f'3mAttackRate_{name}.csv'))

        so = scen_order
        if name == 'no_normal':
            d = d.loc[d['key1']!='as_normal'] # Remove normal
            so = so[1:]

        g = sns.FacetGrid(data=d, row='Group', height=4, aspect=3, row_order=['Teachers & Staff', 'Students'], legend_out=False)

        g.map_dataframe( sns.barplot, x='key1', y='Cum Inc (%)', hue='key2', hue_order=test_order, order=so, palette=test_hue)
        if name == 'all':
            g.add_legend(fontsize=14)

        g.set_titles(row_template="{row_name}")
        xtl = g.axes[1,0].get_xticklabels()
        xtl = [l.get_text() for l in xtl]
        g.set(xticklabels=[scen_names[k] if k in scen_names else k for k in xtl])
        g.set_axis_labels(y_var="3-Month Attack Rate (%)")
        plt.tight_layout()
        cv.savefig(os.path.join(imgdir, f'3mAttackRate_{name}.png'), dpi=300)



#%% Fig. 2 -- Population-wide reproduction number

if to_plot['Fig. 2']:
    df.to_csv(os.path.join(imgdir, f'Re_3mAverage.csv'))
    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.set_window_title('Fig. 2: Population-wide reproduction number')
    sns.barplot(data=df, x='key1', y='re', hue='key2', hue_order=test_order, order=scen_order, palette=test_hue)
    ax.set_ylim([0.8, 1.45])
    ax.set_ylabel(r'Population-wide reproduction number ($R_e$)')
    ax.set_xlabel('')
    xtl = ax.get_xticklabels()
    xtl = [l.get_text() for l in xtl]
    ax.set_xticklabels([scen_names[k] if k in scen_names else k for k in xtl])
    ax.axhline(y=1, color='k', ls=':', lw=2)
    plt.legend().set_title('')
    plt.tight_layout()
    cv.savefig(os.path.join(imgdir, f'Re_3mAverage_{figsize}.png'), dpi=300)



#%% Fig. 4 -- Fraction of in-person days lost

if to_plot['Fig. 4']:
    d = pd.melt(df, id_vars=['key1', 'key2'], value_vars=[f'perc_inperson_days_lost_{gkey}' for gkey in grp_dict.keys()], var_name='Group', value_name='Days lost (%)')
    d.replace( {'Group': {f'perc_inperson_days_lost_{gkey}':gkey for gkey in grp_dict.keys()}}, inplace=True)
    d.to_csv(os.path.join(imgdir, f'InPersonDaysLost.csv'))
    g = sns.FacetGrid(data=d, row='Group', height=4, aspect=aspect, row_order=['Teachers & Staff', 'Students'], legend_out=False)
    g.map_dataframe( sns.barplot, x='key1', y='Days lost (%)', hue='key2', hue_order=test_order, order=scen_order, palette=test_hue) #'Reds'
    g.add_legend(fontsize=14)
    g.set_titles(row_template="{row_name}", fontsize=24)
    xtl = g.axes[1,0].get_xticklabels()
    xtl = [l.get_text() for l in xtl]
    g.set(xticklabels=[scen_names[k] if k in scen_names else k for k in xtl])
    g.set_axis_labels(y_var="Remote learning days (%)")
    plt.tight_layout()
    plt.gcf().canvas.set_window_title('Fig. 4: Remote learning days')
    cv.savefig(os.path.join(imgdir, f'InPersonDaysLost_{aspect}.png'), dpi=300)



#%% Number of diagnostic tests required

if to_plot['Additional tests']:

    fig, ax = plt.subplots(figsize=(12,8))
    d = pd.DataFrame(byschool)

    # Additional tests per 100,000 population
    d['PCR'] *= 100000 / d['Days'] / d['Pop*Scale']
    d['Antigen'] *= 100000 / d['Days'] / d['Pop*Scale']
    test_order.reverse()

    d['Total'] = d['PCR'] + d['Antigen']
    d = d.loc[d['key1'] == 'with_countermeasures']
    d = d.groupby('key2').mean().loc[test_order][['n', 'PCR', 'Antigen', 'Total']].reset_index()
    ax.barh(d['key2'], d['Total'], color='r', label='Antigen')
    ax.barh(d['key2'], d['PCR'], color='b', label='PCR')
    ax.grid(axis='x')
    ax.set_xlabel('Additional tests required (daily per 100k population)')
    fig.tight_layout()
    plt.legend()
    cv.savefig(os.path.join(imgdir, f'NumTests.png'), dpi=300)

    # Save mean to CSV
    df.groupby(['key1', 'key2']).mean().to_csv(os.path.join(imgdir, 'Mean.csv'))


# Wrap up
sc.toc(T)
print('Done.')
