'''
Simple script to load in population files and print some useful information about schools
'''

import sciris as sc

files = sc.getfilelist('v20201019/inputs')
n_files = len(files)

pops = []
for fn in files:
    print(f'Working on {fn}...')
    pops.append(sc.loadobj(fn))

school_ids = []
school_ids_type = []

# Main loop over population files
for p,pop in enumerate(pops):

    print(f'Pop {p} has {sum(pop.student_flag)} students, {sum(pop.teacher_flag)} teachers, and {sum(pop.staff_flag)} staff')

    students = {}
    teachers = {}
    staff = {}
    for st, sids in pop.school_types.items():
        if st in ['pk', 'uv']:
            continue    # Skip pre-K and universities
        students[st] = 0
        teachers[st] = 0
        staff[st] = 0

        # Loop over each school and add to counts
        for sid in sids:
            uids = pop.schools[sid]
            students[st] += len([u for u in uids if pop.student_flag[u]])
            teachers[st] += len([u for u in uids if pop.teacher_flag[u]])
            staff[st] += len([u for u in uids if pop.staff_flag[u]])

    print(f'Students: {students}')
    print(f'Teachers: {teachers}')
    print(f'Staff: {staff}')

    school_ids_type.append(sc.objdict())
    inds = [i for i,val in enumerate(pop.school_id) if val is not None]
    ids = set(pop.school_id[inds])
    for ind in inds:
        kind = pop.school_type_by_person[ind]
        sid = pop.school_id[ind]
        if kind not in school_ids_type[-1]:
            school_ids_type[-1][kind] = {sid:1}
        elif sid not in school_ids_type[-1][kind]:
            school_ids_type[-1][kind][sid] = 1
        else:
            school_ids_type[-1][kind][sid] += 1
    for k in school_ids_type[-1].keys():
        school_ids_type[-1][k] = len(school_ids_type[-1][k])

    print(f'School {files[p]} has {len(ids)} unique schools')
    print(school_ids_type[-1])
    print(school_ids_type[-1]['es'] + school_ids_type[-1]['ms'] + school_ids_type[-1]['hs'])
    school_ids.append(ids)
