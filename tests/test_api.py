"""API 端到端测试：正常求解、无解、入参校验。"""
from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SOLVED_PAYLOAD = {
    "predictions": [
        {"id": 1, "box": [0, 0, 20, 20], "score": 0.90},
        {"id": 2, "box": [6, 0, 26, 20], "score": 0.85},
        {"id": 3, "box": [100, 100, 120, 120], "score": 0.40},
    ],
    "ground_truths": [
        {"id": 101, "box": [0, 0, 20, 20]},
        {"id": 102, "box": [6, 0, 26, 20]},
    ],
    "target_gt_ids": [102],
    "iou_threshold": 0.5,
    "exemptible_pairs": [[1, 2]],
}


def post(payload):
    return client.post("/api/v1/review", json=payload)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_review_solved():
    resp = post(SOLVED_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "solved"
    assert body["min_exemption_count"] == 1
    assert body["chosen_exempt_pairs"] == [[1, 2]]
    # 基线：GT102 漏检，P3 是误报
    assert body["baseline"]["missed_gt_ids"] == [102]
    assert body["baseline"]["false_positive_prediction_ids"] == [3]
    assert body["baseline"]["unrecovered_target_gt_ids"] == [102]
    # 豁免后：GT102 找回，原有命中保留，误报数不变
    assert body["after"]["hit_gt_ids"] == [101, 102]
    assert body["after"]["recovered_target_gt_ids"] == [102]
    assert body["after"]["false_positive_count"] == body["baseline"]["false_positive_count"]
    assert body["explanation"]


def test_review_already_satisfied():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["target_gt_ids"] = [101]  # 基线已命中
    body = post(payload).json()
    assert body["status"] == "solved"
    assert body["min_exemption_count"] == 0
    assert body["chosen_exempt_pairs"] == []


def test_review_infeasible():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["ground_truths"].append({"id": 103, "box": [200, 200, 220, 220]})
    payload["target_gt_ids"] = [102, 103]  # 103 无任何预测框覆盖
    resp = post(payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "infeasible"
    assert body["min_exemption_count"] is None
    assert body["after"] is None
    assert body["all_exempted"] is not None
    assert body["all_exempted"]["unrecovered_target_gt_ids"] == [103]
    assert "无解" in body["explanation"]
    assert "103" in body["explanation"]


def test_review_ineffective_pair_reported():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["exemptible_pairs"] = [[1, 3]]  # IoU=0，不超过阈值
    body = post(payload).json()
    assert body["ignored_ineffective_pairs"] == [[1, 3]]
    assert body["status"] == "infeasible"  # 没有有效对可找回 102


# ---------------- 入参校验（422） ----------------

def test_too_many_predictions_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["predictions"] = [
        {"id": i, "box": [i * 10, 0, i * 10 + 5, 5], "score": 0.5}
        for i in range(61)
    ]
    payload["exemptible_pairs"] = []
    payload["target_gt_ids"] = []
    assert post(payload).status_code == 422


def test_too_many_ground_truths_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["ground_truths"] = [
        {"id": i, "box": [i * 10, 0, i * 10 + 5, 5]} for i in range(61)
    ]
    payload["target_gt_ids"] = []
    assert post(payload).status_code == 422


def test_too_many_pairs_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["exemptible_pairs"] = [[1, 2]] * 11
    assert post(payload).status_code == 422


def test_unknown_target_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["target_gt_ids"] = [999]
    resp = post(payload)
    assert resp.status_code == 422
    assert "999" in resp.text


def test_unknown_pair_endpoint_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["exemptible_pairs"] = [[1, 99]]
    assert post(payload).status_code == 422


def test_degenerate_box_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["predictions"][0]["box"] = [10, 10, 10, 20]  # x2 == x1
    assert post(payload).status_code == 422


def test_duplicate_prediction_ids_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["predictions"].append({"id": 1, "box": [50, 50, 60, 60], "score": 0.3})
    assert post(payload).status_code == 422


def test_score_out_of_range_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["predictions"][0]["score"] = 1.5
    assert post(payload).status_code == 422


def test_pair_with_same_endpoint_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["exemptible_pairs"] = [[1, 1]]
    assert post(payload).status_code == 422


def test_threshold_out_of_range_rejected():
    payload = copy.deepcopy(SOLVED_PAYLOAD)
    payload["iou_threshold"] = 1.2
    assert post(payload).status_code == 422
