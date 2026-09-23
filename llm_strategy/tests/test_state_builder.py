"""StateBuilder tests: information boundary and fact-only output."""

from __future__ import annotations

import json
import unittest

from personal_train.llm_strategy.state_builder import build_battle_state
from personal_train.llm_strategy.tests.fakes import (
    default_rules,
    make_detection,
    make_detection_dict,
    make_init_obs,
    make_isolated_obs,
)

INIT_CATALOGUE = make_init_obs(
    [
        {"id": 51, "type": 9400, "lon": 123.25, "lat": 24.75, "health": 32.0},
        {"id": 52, "type": 9400, "lon": 123.6, "lat": 23.0, "health": 32.0},
        {"id": 53, "type": 9400, "lon": 123.25, "lat": 21.25, "health": 32.0},
        {"id": 54, "type": 9600, "lon": 119.95, "lat": 23.0, "health": 20.0},
        {"id": 106, "type": 9600, "lon": 121.55, "lat": 23.0, "health": 20.0},
    ]
)


def platforms(step: int = 0, detections=()):
    return [
        make_isolated_obs(
            entity_id=10,
            entity_type=21000,
            step=step,
            lon=117.0,
            lat=22.5,
            detections=detections,
            name="9500_暗巢_误导名",
        ),
        make_isolated_obs(
            entity_id=11, entity_type=21001, step=step, lon=117.1, lat=22.4
        ),
        make_isolated_obs(
            entity_id=12, entity_type=21002, step=step, lon=117.2, lat=22.3
        ),
    ]


class StateBuilderTests(unittest.TestCase):
    def test_known_targets_come_from_the_init_catalogue(self):
        state = build_battle_state(
            step=0,
            init_ship_observation=INIT_CATALOGUE,
            platform_observations=platforms(),
            rules=default_rules(),
        )
        ids = {item["entity_id"] for item in state["known_targets"]}
        self.assertEqual(ids, {51, 52, 53, 54, 106})
        types = {item["type"] for item in state["known_targets"]}
        self.assertEqual(types, {9400, 9600})
        for item in state["known_targets"]:
            self.assertIn(item["type"], (9400, 9600))

    def test_hidden_9500_never_leaks_into_the_state(self):
        # No red detection of 9500 exists; the state must not contain any
        # 9500 record even though the (never-read) global observation would.
        state = build_battle_state(
            step=0,
            init_ship_observation=INIT_CATALOGUE,
            platform_observations=platforms(),
            rules=default_rules(),
        )
        text = json.dumps(state, ensure_ascii=False)
        self.assertNotIn('"entity_type": 9500', text)
        self.assertNotIn('"type": 9500', text)
        self.assertEqual(state["detected_tracks"], [])

    def test_9500_appears_only_after_a_legal_detection(self):
        ship_detection = make_detection(
            entity_id=168, entity_type=9500, lon=119.3, lat=25.35, time=5
        )
        before = build_battle_state(
            step=0,
            init_ship_observation=INIT_CATALOGUE,
            platform_observations=platforms(),
            rules=default_rules(),
        )
        after = build_battle_state(
            step=5,
            init_ship_observation=INIT_CATALOGUE,
            platform_observations=platforms(step=5, detections=[ship_detection]),
            rules=default_rules(),
        )
        self.assertEqual(before["detected_tracks"], [])
        tracks = {item["entity_id"]: item for item in after["detected_tracks"]}
        self.assertIn(168, tracks)
        self.assertEqual(tracks[168]["entity_type"], 9500)
        # Detections never mutate the opening catalogue section.
        self.assertNotIn(
            168, {item["entity_id"] for item in after["known_targets"]}
        )

    def test_misleading_name_chn_does_not_change_type_recognition(self):
        # The 21000 platform carries a name claiming to be a 9500 ship.
        state = build_battle_state(
            step=0,
            init_ship_observation=INIT_CATALOGUE,
            platform_observations=platforms(),
            rules=default_rules(),
        )
        by_id = {item["entity_id"]: item for item in state["red_platforms"]}
        self.assertEqual(by_id[10]["type"], 21000)
        self.assertEqual(by_id[11]["type"], 21001)
        self.assertEqual(by_id[12]["type"], 21002)
        for item in state["red_platforms"]:
            self.assertIn(item["type"], (21000, 21001, 21002))

    def test_detect_info_as_dataclass_and_dict_produce_the_same_tracks(self):
        dataclass_based = build_battle_state(
            step=3,
            init_ship_observation=INIT_CATALOGUE,
            platform_observations=platforms(
                step=3,
                detections=[
                    make_detection(
                        entity_id=168, entity_type=9500, lon=119.3, lat=25.35, time=3
                    )
                ],
            ),
            rules=default_rules(),
        )
        dict_based = build_battle_state(
            step=3,
            init_ship_observation=INIT_CATALOGUE,
            platform_observations=platforms(
                step=3,
                detections=[
                    make_detection_dict(
                        entity_id=168, entity_type=9500, lon=119.3, lat=25.35, time=3
                    )
                ],
            ),
            rules=default_rules(),
        )
        self.assertEqual(dataclass_based["detected_tracks"], dict_based["detected_tracks"])

    def test_state_is_fact_only_and_json_serialisable(self):
        state = build_battle_state(
            step=0,
            init_ship_observation=INIT_CATALOGUE,
            platform_observations=platforms(),
            rules=default_rules(),
        )
        for key in (
            "step",
            "max_steps",
            "red_platforms",
            "known_targets",
            "detected_tracks",
            "satellite_status",
            "satellite_rules",
            "map_rules",
            "weapon_rules",
            "scoring_rules",
        ):
            self.assertIn(key, state)
        # No advisory/priority fields may exist.
        forbidden = {"priority", "recommended", "best_target", "waves", "groups"}
        text = json.dumps(state, ensure_ascii=False)
        for word in forbidden:
            self.assertNotIn(word, text)
        json.loads(text)  # must round-trip

    def test_weapon_and_scoring_rules_match_engine_tables(self):
        state = build_battle_state(
            step=0,
            init_ship_observation=INIT_CATALOGUE,
            platform_observations=platforms(),
            rules=default_rules(),
        )
        hit = state["weapon_rules"]["base_hit_rate"]
        self.assertEqual(hit["21000"]["9400"], 0.8)
        self.assertEqual(hit["21000"]["9600"], 0.6)
        self.assertEqual(hit["21000"]["9500"], 0.0)
        self.assertEqual(hit["21002"]["9500"], 0.8)
        damage = state["weapon_rules"]["base_damage_points"]
        self.assertEqual(damage["21001"]["9400"], 5)
        self.assertEqual(damage["21002"]["9400"], 0)
        weights = state["scoring_rules"]["objective_value_weights"]
        self.assertEqual(
            weights, {"9400": 5.0, "9600": 2.0, "9500": 1.0}
        )


if __name__ == "__main__":
    unittest.main()
