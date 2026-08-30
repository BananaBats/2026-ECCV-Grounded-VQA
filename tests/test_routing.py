#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "routing"))

from route_predictions import APPEAR_TWICE, is_alternate_route, route  # noqa: E402
from validate_submission import validate  # noqa: E402


def track(object_id: int, x0: float) -> dict:
    return {
        "id": object_id,
        "frame_ids": [0],
        "bounding_boxes": [[x0, 0.1, x0 + 0.1, 0.2]],
    }


class RoutingTest(unittest.TestCase):
    def test_rule(self) -> None:
        self.assertTrue(is_alternate_route({"question": "Track both tools.", "targets": [{}, {}]}))
        self.assertFalse(is_alternate_route({"question": "Track one tool.", "targets": [{}]}))
        self.assertFalse(is_alternate_route({"question": "An occlusion game question", "targets": [{}, {}]}))
        self.assertFalse(is_alternate_route({"question": APPEAR_TWICE, "targets": [{}, {}]}))

    def test_route_and_validate(self) -> None:
        base = {"video_1": {"grounded_question": {"1": [track(0, 0.1)]}}}
        alternate = {"video_1": {"grounded_question": {"1": [track(0, 0.3)]}}}
        with tempfile.TemporaryDirectory() as directory:
            plans = Path(directory)
            sample = plans / "video_1_q1"
            sample.mkdir()
            (sample / "multi_anchor_plan.json").write_text(
                '{"question":"Track both tools.","targets":[{},{}]}', encoding="utf-8"
            )
            output, routed, kept, missing = route(
                base, alternate, [("video_1", "1")], plans, allow_missing_alternate=False
            )
        self.assertEqual(routed, ["video_1_q1"])
        self.assertEqual(kept, [])
        self.assertEqual(missing, [])
        self.assertEqual(output["video_1"]["grounded_question"]["1"][0]["bounding_boxes"][0][0], 0.3)
        self.assertEqual(validate(output)["questions"], 1)


if __name__ == "__main__":
    unittest.main()
