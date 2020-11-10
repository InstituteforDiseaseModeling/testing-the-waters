'''
Check that days are counted correctly.
'''

import os
import unittest
import sciris as sc
import numpy as np
import covasim_schools as cvsch
from testing_the_waters.create_sim import create_sim as tw_create_sim



class SchedulingTest(unittest.TestCase):

    def setUp(self):
        self.is_debugging = False
        self.folder_name = os.path.join('inputs')
        self.pop_size = 20_000
        self.rand_seed = 1
        if not os.path.isfile(os.path.join(self.folder_name, f'kc_clustered_{self.pop_size}_seed{self.rand_seed}.ppl')):
            cvsch.make_population(
                pop_size=self.pop_size,
                rand_seed=self.rand_seed,
                do_save=True
            )
        self.entry = {
            "index": 1,
            "mismatch": 0.02961049250551955,
            "pars": {
                "pop_infected": 23,
                "change_beta": 0.6019257954058038,
                "symp_prob": 0.09175356222300922
            }
        }
        self.pars = None
        self.sim = None
        self.result = None
        self.school_stats = None
        return

    def tearDown(self):
        return

    # region configure run utils

    @staticmethod
    def save_json(thing, name_root=None):
        filename = f"DEBUG_{name_root}.json"
        sc.savejson(filename, obj=thing, indent=4, sort_keys=True)

    def default_setup(self):
        self.pars = self.entry['pars']
        seed = self.entry['index']
        self.pars['rand_seed'] = seed

    def create_sim(self):
        if self.sim:
            return
        self.sim = tw_create_sim(
            params=self.pars,
            pop_size=int(self.pop_size),
            folder=self.folder_name,
            popfile_stem=f'kc_clustered_{self.pop_size}_seed'
        )
        return

    def run_sim(self, save_sim=False):
        if not self.sim:
            self.create_sim()
        self.sim.run()
        self.result = self.sim.to_json(
            tostring=False)
        if save_sim or self.is_debugging:
            self.save_json(self.result, self.id())
        self.assertIn('results', self.result)
        return

    @staticmethod
    def create_testing_pars(is_antigen=True, perfection=0):
        test_pars = {
            'start_date': '2020-10-26',
            'repeat': 7,
            'groups': ['students', 'teachers', 'staff'],
            'coverage': 1.0,
            'is_antigen': is_antigen
        }
        if is_antigen:  # add antigen params
            test_pars['symp7d_sensitivity'] = 0.971
            test_pars['other_sensitivity'] = 0.9
            test_pars['specificity'] = 0.985
            test_pars['PCR_followup_perc'] = 0.0
            test_pars['PCR_followup_delay'] = 3.0
            if perfection != 0:
                test_pars['symp7d_sensitivity'] = 1.0
                test_pars['other_sensitivity'] = 1.0
                test_pars['specificity'] = 1.0
            if perfection == -1:
                test_pars['symp7d_sensitivity'] = 0.0
                test_pars['other_sensitivity'] = 0.0
        else:  # add pcr params
            test_pars['sensitivity'] = 1.0
            test_pars['delay'] = 1.0
        return test_pars

    def create_scenario_pars(self, test_pars=None, do_nothing=False,
                             perfect_testing=0, schedule='Full'):
        if test_pars is None:
            test_pars = self.create_testing_pars(perfection=perfect_testing)
        scen_pars = {
            'start_day': '2020-11-02',
            'schedule': schedule,
            'screen_prob': 0.9,
            'test_prob': 0.5,  # Amongst those who screen positive
            'screen2pcr': 3,  # Days from screening to receiving PCR results
            'trace_prob': 0.75,  # Fraction of newly diagnosed index cases who are traced
            'quar_prob': 0.75,  # Of those reached by contact tracing, this fraction will quarantine
            'ili_prob': 0.0,  # Daily ili probability
            'beta_s': 0.75,  # NPI
            'testing': test_pars,
        }
        if do_nothing:
            scen_pars['screen_prob'] = 0.0
            scen_pars['test_prob'] = 0.0
            scen_pars['trace_prob'] = 0.0
            scen_pars['quar_prob'] = 0.0
            scen_pars['beta_s'] = 1.0
            scen_pars['testing']['coverage'] = 0.0
        if perfect_testing == -1:
            scen_pars['screen_prob'] = 0.0
            scen_pars['test_prob'] = 0.0
        elif perfect_testing == 1:
            scen_pars['screen_prob'] = 1.0
            scen_pars['test_prob'] = 1.0
        return scen_pars

    def create_all_school_scenarios(self,
                                    do_nothing=False,
                                    es_pars=None,
                                    ms_pars=None,
                                    hs_pars=None):
        if es_pars is None:
            es_pars = self.create_scenario_pars(do_nothing=do_nothing)
        if ms_pars is None:
            ms_pars = self.create_scenario_pars(do_nothing=do_nothing)
        if hs_pars is None:
            hs_pars = self.create_scenario_pars(do_nothing=do_nothing)
        scenario = {
            'es': es_pars,
            'hs': hs_pars,
            'ms': ms_pars,
            'pk': None,
            'uv': None
        }
        return scenario

    # endregion

    # region simulation results support
    def get_full_result_channel(self, channel):
        result_data = self.result['results'][channel]
        return result_data

    def get_day_zero_channel_value(self, channel):
        """

        Args:
            channel: timeseries channel to report ('n_susceptible')

        Returns: day zero value for channel

        """
        result_data = self.get_full_result_channel(channel=channel)
        return result_data[0]

    def get_day_final_channel_value(self, channel):
        channel = self.get_full_result_channel(channel=channel)
        return channel[-1]

    def get_school_stats(self, save=False):
        stats_json = sc.jsonify(self.sim.school_stats)
        if save or self.is_debugging:
            self.save_json(stats_json, f'school_stats_{self.id()}')
        self.school_stats = stats_json
        return

    # endregion

    # region specialized configurations
    def configure_no_interventions_at_all(self):
        self.enable_disable_default_interventions(
            enable_community_ivs=False,
            enable_es_ivs=False,
            enable_ms_ivs=False,
            enable_hs_ivs=False
        )
        return

    def enable_disable_default_interventions(self,
                                             enable_community_ivs=True,
                                             enable_hs_ivs=True,
                                             enable_ms_ivs=True,
                                             enable_es_ivs=True,
                                             perfect_testing=0,
                                             schedules='Full'):
        self.default_setup()
        self.create_sim()
        es_pars = self.create_scenario_pars(do_nothing=not enable_es_ivs,
                                            perfect_testing=perfect_testing,
                                            schedule=schedules)
        ms_pars = self.create_scenario_pars(do_nothing=not enable_ms_ivs,
                                            perfect_testing=perfect_testing,
                                            schedule=schedules)
        hs_pars = self.create_scenario_pars(do_nothing=not enable_hs_ivs,
                                            perfect_testing=perfect_testing,
                                            schedule=schedules)
        scenario = {
            'es': es_pars,
            'ms': ms_pars,
            'hs': hs_pars,
            'pk': None,
            'uv': None
        }
        school_manager = cvsch.schools_manager(scenario=scenario, shrink=False)
        if not enable_community_ivs:
            self.sim['interventions'] = []
        elif perfect_testing == -1:
            # TODO: be smarter than this.
            self.sim['interventions'].remove(self.sim['interventions'][0])

        self.sim['interventions'] += [school_manager]
        return

    def disable_severe_covid(self):
        prob_length = len(self.sim.pars['prognoses']['severe_probs'])
        self.sim.pars['prognoses']['severe_probs'] = np.array([0.0] * prob_length)
        return

    # endregion

    # region results management
    def get_sample_schools(self, school_types_array):
        sm = self.sim.get_interventions(cvsch.schools_manager) # Get the schools manager to pull out schools
        sample_schools = {}
        school_ids = self.school_stats.keys()
        for school_id in school_ids:
            this_school = self.school_stats[school_id]
            school_type = this_school['type']
            if school_type not in sample_schools and school_type in school_types_array:
                this_school['id'] = school_id
                this_school['school_obj'] = [school for school in sm.schools if school.sid == school_id][0]
                sample_schools[school_type] = this_school
            if len(sample_schools) == len(school_types_array):
                break
        return sample_schools

    def check_attendance_for_school(
            self,
            school,
            school_type,
            expect_full_attendance=True,
            schedule_type='Full'
    ):
        # TODO: change expect_full_attendance to schedule_type
        # TODO: support hybrid for staff, teachers, students
        for role in ['staff', 'teachers', 'students']:
            expected_attendance = school['scheduled'][role]
            found_attendance = school['in_person'][role]
            message_stem = f"\nFor school type: {school_type}, id: {school['id']}, role: {role} expected "
            if expect_full_attendance:
                self.assertEqual(expected_attendance, found_attendance,
                                 msg=message_stem + f"scheduled days {(expected_attendance)} "
                                                    f"to equal actual attendance {(found_attendance)}")
            else:
                self.assertGreater(expected_attendance, found_attendance,
                                   msg=message_stem +
                                   f"scheduled days {(expected_attendance)} "
                                   f"to exceed actual attendance {(found_attendance)}")
            if schedule_type == 'Hybrid':
                # NOTE: this breaks if we run out of a role on a day,
                # for example if all staff are out sick
                staff_scheduled = school['school_obj'].stats.scheduled['staff']
                scheduled_day_index = 0
                consecutive_closed = 0
                group_a_attended_yesterday = False
                group_b_attended_yesterday = False
                school_closed_days = []
                group_a_days = []
                group_b_days = []

                # FIRST: use staff scheduled to make A and B groups
                for scheduled_count_for_day in staff_scheduled:
                    if scheduled_count_for_day == 0:
                        school_closed_days.append(scheduled_day_index)
                        consecutive_closed += 1
                    elif consecutive_closed > 1:  # start of school or end of weekend
                        group_a_days.append(scheduled_day_index)
                        group_a_attended_yesterday = True
                        consecutive_closed = 0
                    elif group_a_attended_yesterday:
                        group_a_days.append(scheduled_day_index)
                        group_a_attended_yesterday = False
                    elif consecutive_closed == 1:  # We are in school and today is thursday
                        group_b_days.append(scheduled_day_index)
                        group_b_attended_yesterday = True
                        consecutive_closed = 0
                    elif group_b_attended_yesterday:
                        group_b_days.append(scheduled_day_index)
                        group_b_attended_yesterday = False
                    scheduled_day_index += 1

                # SECOND: use A and B groups to get attendance by group
                student_in_person = school['school_obj'].stats.in_person['students']
                group_a_attendance = []
                group_b_attendance = []
                attendance_day = 0
                while attendance_day < scheduled_day_index:
                    if attendance_day in school_closed_days:
                        group_a_attendance.append(student_in_person[attendance_day])
                        group_b_attendance.append(student_in_person[attendance_day])
                    elif attendance_day in group_a_days:
                        group_a_attendance.append(student_in_person[attendance_day])
                        group_b_attendance.append(0)
                    elif attendance_day in group_b_days:
                        group_a_attendance.append(0)
                        group_b_attendance.append(student_in_person[attendance_day])
                    attendance_day += 1

                if self.is_debugging:
                    attendance_object = {'staff_scheduled': staff_scheduled,
                                         'a_days': group_a_days,
                                         'b_days': group_b_days,
                                         'c_days_closed': school_closed_days,
                                         'group_a_attendance': group_a_attendance,
                                         'group_b_attendance': group_b_attendance}
                    self.save_json(attendance_object, name_root="attendance_object")

                # THIRD: verify A and B group attendance
                if expect_full_attendance:  # We expect the same attendance every day
                    a_group_expected = group_a_attendance[group_a_days[0]]
                    b_group_expected = group_b_attendance[group_b_days[0]]
                else:  # We expect someone to show up
                    a_group_expected = 1
                    b_group_expected = 1
                student_observed_day = 0
                while student_observed_day < scheduled_day_index:
                    if student_observed_day in school_closed_days:
                        a_expected = 0
                        b_expected = 0
                    elif student_observed_day in group_a_days:
                        a_expected = a_group_expected
                        b_expected = 0
                    else:
                        a_expected = 0
                        b_expected = b_group_expected
                    observed_a = group_a_attendance[student_observed_day]
                    self.assertGreaterEqual(observed_a,
                                            a_expected,
                                            msg=f"On day {student_observed_day}, "
                                                f"expected group a attendance: {a_expected}, "
                                                f"got: {observed_a}")
                    observed_b = group_b_attendance[student_observed_day]
                    self.assertGreaterEqual(observed_b,
                                            b_expected,
                                            msg=f"On day {student_observed_day}, "
                                                f"expected group b attendance: {b_expected}, "
                                                f"got: {observed_b}")
                    student_observed_day += 1
        return

    def verify_no_positive_tests(self):
        # verify testing but with no interventions
        final_tests = self.get_day_final_channel_value("cum_tests")
        final_quarantined = self.get_day_final_channel_value("cum_quarantined")
        final_diagnosed = self.get_day_final_channel_value("cum_diagnoses")
        self.assertGreater(final_tests, 0)
        self.assertEqual(0, final_diagnosed)
        self.assertEqual(0, final_quarantined)
        return

    # endregion

    def test_schools_manager_no_infections(self):
        """
        Test shows that with no infections, and 100% bulletproof testing,
        there are no diagnoses, no quarantined, perfect attendance
        """
        self.is_debugging = False
        self.entry['pars']['pop_infected'] = 0
        self.enable_disable_default_interventions(perfect_testing=1)
        self.run_sim()

        # verify testing happened but no interventions took place
        self.verify_no_positive_tests()

        # verify school stats working for no interventions
        self.get_school_stats()

        # For each school type verify full attendance
        untested_school_types = ['pk', 'es', 'ms', 'hs', 'uv']
        sample_schools = self.get_sample_schools(untested_school_types)
        for school_type in sample_schools:
            school_in_question = sample_schools[school_type]
            # see that everyone attends all the time
            self.check_attendance_for_school(
                school=school_in_question, school_type=school_type, expect_full_attendance=True
            )
        return

    def test_schools_manager_no_diagnoses(self):
        """
        Test shows that with testing that at 0 sensitivity and 1 specificity,
        and no severe covid there are no diagnoses, quarantine, or lost days
        """
        self.enable_disable_default_interventions(
            perfect_testing=-1,
            enable_community_ivs=False
        )
        self.disable_severe_covid()
        self.run_sim()

        self.get_school_stats()

        # verify perfect attendance with no successful diagnoses
        schools_to_sample = ['es', 'ms', 'hs']
        sample_schools = self.get_sample_schools(school_types_array=schools_to_sample)
        for school in schools_to_sample:
            # First, ensure that there are infectious people in attendance
            for role in ['staff', 'teachers', 'students']:
                self.assertGreater((sample_schools[school]['infectious_stay_at_school'][role]),
                                   0)
            # Second, ensure that they actually attended
            self.check_attendance_for_school(
                school=sample_schools[school],
                school_type=school,
                expect_full_attendance=True
            )
        return

    def test_no_diagnoses_except_ms(self):
        self.enable_disable_default_interventions(enable_community_ivs=False,
                                                  enable_hs_ivs=False,
                                                  enable_ms_ivs=True,
                                                  enable_es_ivs=False)
        self.disable_severe_covid()
        self.run_sim()
        self.get_school_stats()
        schools_to_sample = ['es', 'ms', 'hs']
        sample_schools = self.get_sample_schools(school_types_array=schools_to_sample)

        # Check that cvsch without tests lost no days
        untested_schools = ['es', 'hs']
        for school in untested_schools:
            self.check_attendance_for_school(
                school=sample_schools[school],
                school_type=school,
                expect_full_attendance=True
            )

        # Check that school with testing lost days
        self.check_attendance_for_school(school=sample_schools['ms'],
                                         school_type='ms',
                                         expect_full_attendance=False)
        return

    def test_no_diagnoses_except_hs(self):
        self.enable_disable_default_interventions(enable_community_ivs=False,
                                                  enable_hs_ivs=True,
                                                  enable_ms_ivs=False,
                                                  enable_es_ivs=False)
        self.disable_severe_covid()
        self.run_sim()
        self.get_school_stats()
        schools_to_sample = ['es', 'ms', 'hs']
        sample_schools = self.get_sample_schools(school_types_array=schools_to_sample)

        # Check that cvsch without tests lost no days
        untested_schools = ['es', 'ms']
        for school in untested_schools:
            self.check_attendance_for_school(
                school=sample_schools[school],
                school_type=school,
                expect_full_attendance=True
            )

        # Check that school with testing lost days
        self.check_attendance_for_school(school=sample_schools['hs'],
                                         school_type='hs',
                                         expect_full_attendance=False)
        return

    def test_no_diagnoses_except_es(self):
        self.enable_disable_default_interventions(enable_community_ivs=False,
                                                  enable_hs_ivs=False,
                                                  enable_ms_ivs=False,
                                                  enable_es_ivs=True)
        self.disable_severe_covid()
        self.run_sim()
        self.get_school_stats()
        schools_to_sample = ['es', 'ms', 'hs']
        sample_schools = self.get_sample_schools(school_types_array=schools_to_sample)

        # Check that cvsch without tests lost no days
        untested_schools = ['ms', 'hs']
        for school in untested_schools:
            self.check_attendance_for_school(
                school=sample_schools[school],
                school_type=school,
                expect_full_attendance=True
            )

        # Check that school with testing lost days
        self.check_attendance_for_school(school=sample_schools['es'],
                                         school_type='es',
                                         expect_full_attendance=False)
        return

    def test_schedule_hybrid(self):
        """
        Test shows that with no infections, and 100% bulletproof testing,
        there are no diagnoses, no quarantined, perfect attendance
        """
        self.is_debugging = False
        self.entry['pars']['pop_infected'] = 0
        self.enable_disable_default_interventions(perfect_testing=1,
                                                  schedules='Hybrid')
        self.run_sim()

        self.verify_no_positive_tests()

        # verify school stats working for no interventions
        self.get_school_stats()

        # For each school type verify full attendance
        untested_school_types = ['pk', 'es', 'ms', 'hs', 'uv']
        sample_schools = self.get_sample_schools(untested_school_types)
        for school_type in sample_schools:
            school_in_question = sample_schools[school_type]
            # see that everyone attends all the time
            self.check_attendance_for_school(
                school=school_in_question, school_type=school_type, expect_full_attendance=True,
                schedule_type='Hybrid'
            )
        return


if __name__ == '__main__':
    unittest.main()
