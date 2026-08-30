#!/usr/bin/env python3
from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "sparse_fusion"))

from fuse_dense_sparse import (  # noqa: E402
    fuse_sample,
    interpolate_track,
    recover_sampled_sam_tracks,
    style_couple,
    validate_official_sample,
)
from generate_sparse import (  # noqa: E402
    SparseFrame,
    SparseObject,
    SparseResponse,
    build_prompt,
    validate_response,
)
from sparse_common import (  # noqa: E402
    box_iou,
    expected_ids,
    save_json,
)


class CommonTest(unittest.TestCase):
    def test_iou(self) -> None:
        self.assertAlmostEqual(box_iou([0, 0, 100, 100], [0, 0, 100, 100]), 1.0)

    def test_plan_support_and_response_validation(self) -> None:
        plan = {"targets": [
            {"obj_id": 0, "visibility_segments": [{"first_frame_idx": 0, "last_frame_idx": 60}]},
            {"obj_id": 1, "visibility_segments": [{"first_frame_idx": 30, "last_frame_idx": 90}]},
        ]}
        self.assertEqual(expected_ids(plan, 0), {0})
        self.assertEqual(expected_ids(plan, 30), {0, 1})
        response = SparseResponse(frames=[
            SparseFrame(frame_idx=0, objects=[SparseObject(
                object_id=0, visibility="visible", box_2d=[1, 2, 30, 40], confidence=.9
            )]),
            SparseFrame(frame_idx=30, objects=[
                SparseObject(object_id=0, visibility="visible", box_2d=[1, 2, 30, 40], confidence=.9),
                SparseObject(object_id=1, visibility="fully_occluded", box_2d=[5, 6, 20, 25], confidence=.5),
            ]),
        ])
        self.assertEqual(len(validate_response(response, plan, [0, 30])), 3)

    def test_occlusion_gap_allows_hidden_or_absent_target(self) -> None:
        plan = {"targets": [{
            "obj_id": 0,
            "visibility_segments": [
                {"first_frame_idx": 0, "last_frame_idx": 0},
                {"first_frame_idx": 60, "last_frame_idx": 60},
            ],
        }]}
        hidden = SparseResponse(frames=[SparseFrame(frame_idx=30, objects=[
            SparseObject(
                object_id=0, visibility="fully_occluded",
                box_2d=[10, 20, 40, 50], confidence=.7,
            )
        ])])
        absent = SparseResponse(frames=[SparseFrame(frame_idx=30, objects=[])])
        self.assertEqual(len(validate_response(hidden, plan, [30])), 1)
        self.assertEqual(validate_response(absent, plan, [30]), [])
        prompt = build_prompt(plan, [30], {"width": 100, "height": 100})
        self.assertIn("high-priority occlusion interval", prompt)
        self.assertIn("omit it", prompt)

    def test_invalid_response_missing_id(self) -> None:
        plan = {"targets": [
            {"obj_id": 0, "visibility_segments": []},
            {"obj_id": 1, "visibility_segments": []},
        ]}
        response = SparseResponse(frames=[SparseFrame(frame_idx=0, objects=[
            SparseObject(object_id=0, visibility="visible", box_2d=[1, 2, 30, 40], confidence=.9)
        ])])
        with self.assertRaisesRegex(ValueError, "misses visibly confirmed IDs"):
            validate_response(response, plan, [0])


class CouplingTest(unittest.TestCase):
    def test_style_residual_interpolates(self) -> None:
        dense = {frame: [100.0, 100.0, 200.0, 200.0] for frame in range(61)}
        sparse = {
            0: [90.0, 90.0, 210.0, 210.0],
            60: [80.0, 80.0, 220.0, 220.0],
        }
        styled, anchors = style_couple(dense, sparse)
        self.assertEqual(anchors, [0, 60])
        self.assertEqual(styled[0], sparse[0])
        self.assertEqual(styled[60], sparse[60])
        self.assertEqual(styled[30], [85.0, 85.0, 215.0, 215.0])


    def test_v13_conditional_gap_interpolation(self) -> None:
        moving = {0: [0, 0, 100, 100], 30: [500, 500, 600, 600]}
        stable = {0: [0, 0, 100, 100], 30: [5, 5, 105, 105]}
        self.assertEqual(sorted(interpolate_track(moving, 6)), [0, 30])
        self.assertEqual(len(interpolate_track(stable, 6)), 31)

    def test_compact_sam_coverage_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            dense_dir = Path(tmp_name)
            save_json(dense_dir / "run_result.json", {
                "non_empty_frames": [0, 3, 6, 9, 12],
                "segment_gating": {"intervals": {"0": [[0, 6]]}},
            })
            dense = {0: {frame: [0, 0, 10, 10] for frame in range(13)}}
            recovered, meta = recover_sampled_sam_tracks(dense_dir, dense)
            self.assertEqual(sorted(recovered[0]), [0, 3, 6])
            self.assertTrue(meta["lossy"])


    def test_inhong_v13_reference_function_parity(self) -> None:
        source = (
            HERE.parent / "references" / "inhong_v13" / "fuse_v13.py"
        ).read_text()
        tree = ast.parse(source)
        wanted = {"style_couple", "_sane", "interpolate_track"}
        functions = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in wanted
        ]
        namespace = {"ft": type("FT", (), {"iou": staticmethod(box_iou)})()}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "v13_reference", "exec"), namespace)
        dense = {frame: [100.0 + frame, 100.0, 200.0 + frame, 200.0] for frame in range(0, 91, 3)}
        sparse = {
            0: [90.0, 90.0, 210.0, 210.0],
            30: [125.0, 88.0, 235.0, 212.0],
            60: [155.0, 85.0, 265.0, 215.0],
            90: [185.0, 80.0, 295.0, 220.0],
        }
        ours_styled, _ = style_couple(dense, sparse)
        self.assertEqual(ours_styled, namespace["style_couple"](dense, sparse))
        merged = dict(sparse)
        merged.update(ours_styled)
        self.assertEqual(
            interpolate_track(merged, 6),
            namespace["interpolate_track"](merged, 6),
        )


class EndToEndFusionTest(unittest.TestCase):
    def test_fixture_reconstructs_v13_dense_motion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            sample = "video_1_q4"
            plan_root = root / "plans"
            dense_root = root / "dense"
            sparse_root = root / "sparse"
            output_root = root / "fused"
            plan = {
                "targets": [{"obj_id": 0, "visibility_segments": [{"first_frame_idx": 0, "last_frame_idx": 60}]}],
                "anchors": [{"frame_idx": 30, "boxes": [{"obj_id": 0, "box_2d": [90, 90, 210, 210]}]}],
            }
            save_json(plan_root / sample / "multi_anchor_plan.json", plan)
            dense_track = {
                "id": 0, "score": 1.0, "frame_ids": list(range(61)),
                "bounding_boxes": [[.1, .1, .2, .2] for _ in range(61)],
            }
            dense_payload = {"video_1": {"grounded_question": {"4": [dense_track]}}}
            save_json(dense_root / sample / "predictions.json", dense_payload)
            save_json(dense_root / sample / "run_result.json", {
                "num_video_frames": 61,
                "non_empty_frames": list(range(0, 61, 3)),
                "segment_gating": {"intervals": {"0": [[0, 60]]}},
            })
            sparse_payload = {"predictions_by_frame": {
                str(frame): [{"object_id": 0, "box_2d_yxyx_1000": [90, 90, 210, 210]}]
                for frame in (0, 30, 60)
            }}
            save_json(sparse_root / sample / "sparse_predictions.json", sparse_payload)
            args = Namespace(
                plan_root=plan_root, dense_root=dense_root, sparse_root=sparse_root,
                output_root=output_root, min_anchor_iou=.25, min_shared=3,
                gate_iou=.30, short_gap=6,
            )
            payload, meta = fuse_sample(args, "video_1", "4")
            self.assertTrue(meta["targets"][0]["trusted"])
            self.assertEqual(
                meta["sam_coverage_recovery"]["source"],
                "run_result.non_empty_frames+segment_gating.intervals",
            )
            validate_official_sample(payload, "video_1", "4", 61)
            track = payload["video_1"]["grounded_question"]["4"][0]
            self.assertEqual(track["frame_ids"], list(range(61)))
            self.assertEqual(track["bounding_boxes"][30], [.09, .09, .21, .21])


if __name__ == "__main__":
    unittest.main(verbosity=2)
