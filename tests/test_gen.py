"""Regression tests for the Set 18 optimization pipeline."""

import unittest

import gen


class GeneticPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = gen.load_data(gen.DATA_FILE)
        cls.units = [unit["apiName"] for unit in data["units"] if unit["traits"]]
        cls.traits = gen.process_traits(data["traits"])
        cls.matrix = gen.create_traits_matrix(cls.traits, cls.units)

    def test_all_thresholds_are_positive_integers(self):
        self.assertTrue(
            all(
                isinstance(threshold, int) and threshold > 0
                for trait in self.traits
                for threshold in trait["thresholds"]
            )
        )

    def test_eclipse_has_no_numeric_threshold(self):
        eclipse = next(trait for trait in self.traits if trait["name"] == "Eclipse")
        self.assertEqual(eclipse["thresholds"], [])

    def test_fitness_accepts_set18_population(self):
        population = gen.create_population(10, len(self.units))
        fitness = [
            gen.compute_fitness(individual, self.matrix, self.traits)
            for individual in population
        ]
        self.assertTrue(all(isinstance(score, int) for score in fitness))


if __name__ == "__main__":
    unittest.main()
