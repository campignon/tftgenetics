"""Integrity tests for the committed Set 18 snapshot."""

import json
import unittest
from collections import Counter
from pathlib import Path

from scripts.update_set18_data import validate_snapshot


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "TFTSet18_full_lookup.json"


class Set18DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with DATA_FILE.open("r", encoding="utf-8") as file:
            cls.data = json.load(file)

    def test_snapshot_is_valid(self):
        validate_snapshot(self.data)

    def test_expected_content(self):
        self.assertEqual(len(self.data["units"]), 65)
        self.assertEqual(len(self.data["traits"]), 36)
        self.assertEqual(len(self.data["emblems"]), 20)
        self.assertEqual(
            Counter(unit["cost"] for unit in self.data["units"]),
            Counter({1: 14, 2: 13, 3: 14, 4: 14, 5: 10}),
        )

    def test_set_has_no_trait_variants(self):
        self.assertFalse(any("variants" in trait for trait in self.data["traits"]))

    def test_lux_avatar_choices(self):
        lux = next(unit for unit in self.data["units"] if unit["apiName"] == "DA_Lux18_Base")
        choices = lux["traitChoice"]["options"]
        self.assertEqual(len(choices), 9)
        self.assertTrue(all(choice["count"] == 2 for choice in choices))

    def test_elder_dragon_special_contributions(self):
        elder = next(
            unit
            for unit in self.data["units"]
            if unit["apiName"] == "DA_18_ElderDragon"
        )
        riftbeast = next(trait for trait in elder["traits"] if trait["name"] == "Riftbeast")
        self.assertEqual(elder["teamSlots"], 2)
        self.assertEqual(riftbeast["count"], 2)

    def test_eclipse_is_a_derived_trait(self):
        eclipse = next(trait for trait in self.data["traits"] if trait["name"] == "Eclipse")
        self.assertEqual(
            eclipse["activation"]["traits"],
            ["DA_18_Solar", "DA_18_Lunar"],
        )


if __name__ == "__main__":
    unittest.main()
