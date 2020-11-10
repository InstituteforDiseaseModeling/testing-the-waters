'''
Run tests for different school interventions and check that students are counted
correctly.
'''


from collections import namedtuple
import covasim_schools as cvsch
import os
from pathlib import Path
import sciris as sc
from tempfile import mkstemp
from testing_the_waters.create_sim import create_sim as cs
from testing_the_waters.testing_scenarios import generate_scenarios , generate_testing
import unittest

TestResults = namedtuple("TestResults", ["scheduled", "in_person", "antigen", "pcr"])


class SchoolScheduleTests(unittest.TestCase):

    population_size = 20_000
    population_filename = None  # "e:\src\covid-cvsch\tests\tmp0u9x6q4l.ppl"
    rand_seed = 1
    parameters = {
        "pop_infected": 242.1,
        "change_beta": 0.53139,
        "symp_prob": 0.0825,
        "rand_seed": rand_seed
    }
    baseline = None
    only_screening = None
    screening_and_pcr = None
    screening_and_antigen = None
    all_remote = None

    @classmethod
    def setUpClass(cls):

        if not cls.population_filename:
            # Create a representative population for the individual tests.
            handle, cls.population_filename = mkstemp(".ppl", dir=str(Path(__file__).parent))
            os.close(handle)
            cvsch.make_population(
                pop_size=cls.population_size,
                rand_seed=cls.rand_seed,
                do_save=True,
                popfile=cls.population_filename,
                cohorting=True)

        cls.run_baseline()
        cls.run_full_attendance_screening_only()
        cls.run_full_attendance_screening_and_pcr()
        cls.run_full_attendance_screening_and_antigen()
        cls.run_all_remote()

        return

    def setUp(self):
        # This page intentionally left blank.
        return

    def tearDown(self):
        # This page intentionally left blank.
        return

    @classmethod
    def tearDownClass(cls):
        # This page intentionally left blank.
        return

    # Baseline, cvsch open, no screening or diagnostics.
    # Expect: no days missed, most cases for students, teachers, and staff
    @classmethod
    def run_baseline(cls):

        print("Running baseline scenario ('as_normal') with no testing:")
        simulation = cs(
            cls.parameters,
            pop_size=cls.population_size,
            load_pop=False,
            people=cls.population_filename)

        scenario = generate_scenarios()["as_normal"]
        testing = generate_testing()["None"]
        for setting in scenario.values():       # setting will be one of pk, es, ms, hs, uv
            if setting is not None:             # setting might be none, e.g. no pre-k in the scenario
                setting["testing"] = testing
        school_interventions = cvsch.schools_manager(scenario)
        simulation['interventions'] += [school_interventions]

        sc.tic()
        simulation.run()
        sc.toc()

        results = simulation.school_results

        print(f"Scheduled days {results.scheduled['all']}")
        print(f"In-person days {results.in_person['all']}")

        cls.baseline = TestResults(
            results.scheduled['all'], results.in_person['all'], results.n_tested["Antigen"], results.n_tested["PCR"]
        )

        return

    def test_baseline(self):

        self.assertIsNotNone(self.baseline)

        self.assertEqual(self.baseline.antigen, 0)
        self.assertEqual(self.baseline.pcr, 0)

        return

    # Full attendance + 100% screening (no diagnostics) -> quarantine
    # Expect: mid-level cases, second highest days missed
    @classmethod
    def run_full_attendance_screening_only(cls):

        print("Running screening (no followup)...")

        simulation = cs(
            cls.parameters,
            pop_size=cls.population_size,
            load_pop=False,
            people=cls.population_filename)

        scenario = generate_scenarios()["as_normal"]
        testing = generate_testing()["None"]
        for setting in scenario.values():       # setting will be one of pk, es, ms, hs, uv
            if setting is not None:             # setting might be none, e.g. no pre-k in the scenario
                setting["screen_prob"] = 1.0    # full screening
                setting["testing"] = testing
        school_interventions = cvsch.schools_manager(scenario)
        simulation['interventions'] += [school_interventions]

        sc.tic()
        simulation.run()
        sc.toc()

        results = simulation.school_results

        print(f"Scheduled days {results.scheduled['all']}")
        print(f"In-person days {results.in_person['all']}")

        cls.only_screening = TestResults(
            results.scheduled['all'], results.in_person['all'], results.n_tested["Antigen"], results.n_tested["PCR"]
        )

        return

    def test_full_attendance_screening_only(self):

        self.assertIsNotNone(self.baseline)
        self.assertIsNotNone(self.only_screening)

        self.assertEqual(self.only_screening.antigen, 0)     # diagnostics turned off
        self.assertEqual(self.only_screening.pcr, 0)         # diagnostics turned off

        # screening will send some students home thus fewer in person days
        self.assertLess(self.only_screening.in_person, self.baseline.in_person)

        return

    # Full attendance + 100% screening + PCR diagnostics (quarantine except for PCR -tive for non-COVID ILI)
    # Expect: mid-level cases ≈ screening w/out diagnostics, slightly fewer days missed (ILIs return to school)
    @classmethod
    def run_full_attendance_screening_and_pcr(cls):

        print("Running screening with PCR followup...")

        simulation = cs(
            cls.parameters,
            pop_size=cls.population_size,
            load_pop=False,
            people=cls.population_filename)

        scenario = generate_scenarios()["as_normal"]
        testing = sc.dcp(generate_testing()["PCR every 1d"])
        testing[0]["delay"] = 2
        for setting in scenario.values():       # setting will be one of pk, es, ms, hs, uv
            if setting is not None:             # setting might be none, e.g. no pre-k in the scenario
                setting["screen_prob"] = 1.0    # full screening
                setting["testing"] = testing
        school_interventions = cvsch.schools_manager(scenario)
        simulation['interventions'] += [school_interventions]

        sc.tic()
        simulation.run()
        sc.toc()

        results = simulation.school_results

        print(f"Scheduled days {results.scheduled['all']}")
        print(f"In-person days {results.in_person['all']}")

        cls.screening_and_pcr = TestResults(
            results.scheduled['all'], results.in_person['all'], results.n_tested["Antigen"], results.n_tested["PCR"]
        )

        return

    def test_full_attendance_screening_and_pcr(self):

        self.assertIsNotNone(self.baseline)
        self.assertIsNotNone(self.only_screening)
        self.assertIsNotNone(self.screening_and_pcr)

        self.assertEqual(self.screening_and_pcr.antigen, 0)    # diagnostics turned off
        self.assertGreater(self.screening_and_pcr.pcr, 0)      # pcr testing is _on_

        # screening will send some people home
        self.assertLess(self.screening_and_pcr.in_person, self.baseline.in_person)

        # PCR will identify ili, if present and allow back to school -- but not enough to overcome stochastic effects, so will fail for many seeds
        # self.assertGreater(self.screening_and_pcr.in_person, self.only_screening.in_person)

        return

    # Full attendance + 100% screening + antigen diagnostics (quarantine except for antigen -tive for non-COVID ILI)
    # Expect: mid-level cases ≈ screening w/out diagnostics, slightly _more_ days missed (antigen specificity)
    @classmethod
    def run_full_attendance_screening_and_antigen(cls):

        print("Running screening with antigen followup...")

        simulation = cs(
            cls.parameters,
            pop_size=cls.population_size,
            load_pop=False,
            people=cls.population_filename)

        scenario = generate_scenarios()["as_normal"]
        testing = [{
            'start_date': '2020-10-26',
            'repeat': 1,
            'groups': ['students', 'teachers', 'staff'],
            'coverage': 1,
            'is_antigen': True,
            'symp7d_sensitivity': 0.971,    # https://www.fda.gov/media/141570/download
            'other_sensitivity': 0.90,      # Modeling assumption
            'specificity': 0.985,           # https://www.fda.gov/media/141570/download
            'PCR_followup_perc': 0.0,
            'PCR_followup_delay': 0.0,      # Does not matter with no PCR follow-up
        }]
        for setting in scenario.values():       # setting will be one of pk, es, ms, hs, uv
            if setting is not None:             # setting might be none, e.g. no pre-k in the scenario
                setting["screen_prob"] = 1.0    # full screening
                setting["testing"] = testing
        school_interventions = cvsch.schools_manager(scenario)
        simulation['interventions'] += [school_interventions]

        sc.tic()
        simulation.run()
        sc.toc()

        results = simulation.school_results

        print(f"Scheduled days {results.scheduled['all']}")
        print(f"In-person days {results.in_person['all']}")

        cls.screening_and_antigen = TestResults(
            results.scheduled['all'], results.in_person['all'], results.n_tested["Antigen"], results.n_tested["PCR"]
        )

        return

    def test_full_attendance_screening_and_antigen(self):

        self.assertIsNotNone(self.baseline)
        self.assertIsNotNone(self.only_screening)
        self.assertIsNotNone(self.screening_and_pcr)
        self.assertIsNotNone(self.screening_and_antigen)

        self.assertGreater(self.screening_and_antigen.antigen, 0)    # antigen testing is _on_
        self.assertEqual(self.screening_and_antigen.pcr, 0)          # diagnostics turned off

        # antigen test sends people home and with imperfect specificity, it sends people home erroneously
        self.assertLess(self.screening_and_antigen.in_person, self.baseline.in_person)

        # antigen test sends more people home on false positive than it "recovers" from ili
        self.assertLess(self.screening_and_antigen.in_person, self.only_screening.in_person)

        # antigen test sends more people home on false positive than it "recovers" from ili and pcr is perfect
        self.assertLess(self.screening_and_antigen.in_person, self.screening_and_pcr.in_person)

        return

    # Full remote, no screening or diagnostics.
    # Expect: all days missed, least cases for students, teachers, and staff
    @classmethod
    def run_all_remote(cls):

        print("Running all students remote...")

        simulation = cs(
            cls.parameters,
            pop_size=cls.population_size,
            load_pop=False,
            people=cls.population_filename)

        scenario = generate_scenarios()["all_remote"]
        testing = generate_testing()["None"]
        for setting in scenario.values():       # setting will be one of pk, es, ms, hs, uv
            if setting is not None:             # setting might be none, e.g. no pre-k in the scenario
                setting["testing"] = testing
        school_interventions = cvsch.schools_manager(scenario)
        simulation['interventions'] += [school_interventions]

        sc.tic()
        simulation.run()
        sc.toc()

        results = simulation.school_results

        print(f"Scheduled days {results.scheduled['all']}")
        print(f"In-person days {results.in_person['all']}")

        cls.all_remote = TestResults(
            results.scheduled['all'], results.in_person['all'], results.n_tested["Antigen"], results.n_tested["PCR"]
        )

        return

    def test_all_remote(self):

        self.assertIsNotNone(self.all_remote)

        self.assertEqual(self.all_remote.in_person, 0)   # everyone stayed home
        self.assertEqual(self.all_remote.antigen, 0)     # no one was tested (everyone home)
        self.assertEqual(self.all_remote.pcr, 0)         # no one was tested (everyone home)

        return


if __name__ == "__main__":
    unittest.main()
