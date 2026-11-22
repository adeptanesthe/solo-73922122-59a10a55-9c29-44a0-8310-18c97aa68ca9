"""核心算法单元测试：IoU、NMS、匹配、最少豁免搜索。"""
from __future__ import annotations

import pytest

from app.solver import (
    EvalResult,
    GroundTruth,
    Prediction,
    evaluate,
    iou,
    is_feasible,
    solve,
)

GT = GroundTruth


def P(pid: int, box, score: float) -> Prediction:
    return Prediction(id=pid, box=tuple(box), score=score)


def G(gid: int, box) -> GroundTruth:
    return GroundTruth(id=gid, box=tuple(box))


def run(preds, gts, thr=0.5, pairs=()):
    """便捷入口：直接对给定框集合做一次 NMS+匹配。"""
    n, m = len(preds), len(gts)
    pp = [[iou(preds[i].box, preds[j].box) if i != j else 0.0 for j in range(n)]
          for i in range(n)]
    pg = [[iou(preds[i].box, gts[g].box) for g in range(m)] for i in range(n)]
    exempt = frozenset(frozenset(p) for p in pairs)
    return evaluate(preds, gts, thr, exempt, pp, pg)


# ---------------- IoU ----------------

def test_iou_identical():
    assert iou((0, 0, 2, 2), (0, 0, 2, 2)) == pytest.approx(1.0)


def test_iou_disjoint():
    assert iou((0, 0, 1, 1), (2, 2, 3, 3)) == 0.0


def test_iou_partial():
    # 交集 2，并集 6 → 1/3
    assert iou((0, 0, 2, 2), (1, 0, 3, 2)) == pytest.approx(1 / 3)


def test_iou_contained():
    # 小框完全在大框内：交集 2，并集 4 → 0.5
    assert iou((0, 0, 2, 2), (1, 0, 2, 2)) == pytest.approx(0.5)


# ---------------- NMS ----------------

def test_only_kept_boxes_can_suppress():
    # B 被 A 抑制；C 与 B 的 IoU>0.5 但与 A ≤0.5，B 已 suppressed 不能再抑制 C
    preds = [P(1, (0, 0, 10, 10), 0.9),
             P(2, (1, 0, 11, 10), 0.8),
             P(3, (4, 0, 14, 10), 0.7)]
    res = run(preds, [])
    assert res.kept_ids == [1, 3]
    assert [(s, by) for s, by, _ in res.suppressed] == [(2, 1)]


def test_suppression_requires_iou_strictly_greater_than_threshold():
    # IoU 恰好等于阈值 0.5 → 不抑制
    preds = [P(1, (0, 0, 2, 2), 0.9), P(2, (1, 0, 2, 2), 0.8)]
    res = run(preds, [], thr=0.5)
    assert res.kept_ids == [1, 2]
    assert res.suppressed == []


def test_score_tie_breaks_by_smaller_id():
    preds = [P(2, (0, 0, 10, 10), 0.9), P(1, (0, 0, 10, 10), 0.9)]
    res = run(preds, [])
    assert res.kept_ids == [1]
    assert [(s, by) for s, by, _ in res.suppressed] == [(2, 1)]


# ---------------- 匹配 ----------------

def test_match_floor_is_inclusive():
    preds = [P(1, (0, 0, 2, 2), 0.9), P(2, (100, 100, 101, 101), 0.8)]
    gts = [G(1, (1, 0, 2, 2)),          # 与 P1 的 IoU 恰好 0.5 → 命中
           G(2, (100.6, 100, 101.6, 101))]  # 与 P2 的 IoU 0.25 → 不命中
    res = run(preds, gts)
    assert [(p, g) for p, g, _ in res.matches] == [(1, 1)]
    assert res.false_positive_ids == [2]
    assert res.missed_gt_ids == [2]


def test_match_iou_tie_prefers_smaller_gt_id():
    preds = [P(1, (0, 0, 10, 10), 0.9)]
    gts = [G(5, (0, 0, 10, 10)), G(2, (0, 0, 10, 10))]
    res = run(preds, gts)
    assert res.matches[0][1] == 2
    assert res.missed_gt_ids == [5]


def test_match_is_greedy_by_score_order_and_max_iou():
    preds = [P(1, (0, 0, 10, 10), 0.9), P(2, (20, 0, 30, 10), 0.8)]
    # GT2 故意排在列表前面：P1 仍应取 IoU 最大的 GT1 而非先出现的 GT2
    gts = [G(2, (1, 0, 11, 10)), G(1, (0, 0, 10, 10)), G(3, (20, 0, 30, 10))]
    res = run(preds, gts)
    assert [(p, g) for p, g, _ in res.matches] == [(1, 1), (2, 3)]


# ---------------- 豁免搜索 ----------------

def recovery_case():
    """两个相邻缺陷：P2 命中 GT102 但被 P1 抑制。"""
    preds = [P(1, (0, 0, 20, 20), 0.90),
             P(2, (6, 0, 26, 20), 0.85),
             P(3, (100, 100, 120, 120), 0.40)]
    gts = [G(101, (0, 0, 20, 20)), G(102, (6, 0, 26, 20))]
    return preds, gts


def test_baseline_misses_adjacent_defect():
    preds, gts = recovery_case()
    res = solve(preds, gts, [102], 0.5, [(1, 2)])
    assert res.baseline.kept_ids == [1, 3]
    assert res.baseline.hit_gt_ids == [101]
    assert res.baseline.missed_gt_ids == [102]
    assert res.baseline.false_positive_ids == [3]


def test_single_exemption_recovers_target():
    preds, gts = recovery_case()
    res = solve(preds, gts, [102], 0.5, [(1, 2)])
    assert res.status == "solved"
    assert res.chosen_pairs == [(1, 2)]
    assert res.optimal_solution_count == 1
    after = res.after
    assert after.kept_ids == [1, 2, 3]
    assert [(p, g) for p, g, _ in after.matches] == [(1, 101), (2, 102)]
    # 误报数未增加（1 → 1）
    assert len(after.false_positive_ids) == len(res.baseline.false_positive_ids) == 1


def test_reversed_duplicate_pairs_counted_once():
    preds, gts = recovery_case()
    res = solve(preds, gts, [102], 0.5, [(1, 2), (2, 1)])
    assert res.status == "solved"
    assert res.chosen_pairs == [(1, 2)]
    assert res.optimal_solution_count == 1  # 重复对不重复计数


def cascade_case():
    """A 抑制 C，C 若被释放则抑制 B：找回 GT12 需要两对豁免。"""
    preds = [P(1, (0, 0, 10, 10), 0.95),
             P(3, (3, 0, 13, 10), 0.90),
             P(2, (6, 0, 16, 10), 0.85)]
    gts = [G(11, (0, 0, 10, 10)), G(12, (3, 0, 13, 10)), G(13, (6, 0, 16, 10))]
    return preds, gts


def test_cascade_requires_two_pairs():
    preds, gts = cascade_case()
    res = solve(preds, gts, [12], 0.5, [(1, 3), (2, 3)])
    assert res.status == "solved"
    assert len(res.chosen_pairs) == 2
    assert res.chosen_pairs == [(1, 3), (2, 3)]
    assert res.after.hit_gt_ids == [11, 12, 13]
    assert res.after.false_positive_ids == []


def test_losing_baseline_hit_is_infeasible():
    # 只豁免 (1,3)：C 被释放后抑制 B，原有命中 GT13 丢失 → 无解
    preds, gts = cascade_case()
    res = solve(preds, gts, [12], 0.5, [(1, 3)])
    assert res.status == "infeasible"
    assert res.after is None
    assert res.all_exempted is not None
    assert 13 in res.all_exempted.missed_gt_ids
    assert "13" in res.explanation


def test_already_satisfied_needs_no_exemption():
    preds = [P(1, (0, 0, 10, 10), 0.9)]
    gts = [G(1, (0, 0, 10, 10))]
    res = solve(preds, gts, [1], 0.5, [])
    assert res.status == "solved"
    assert res.chosen_pairs == []
    assert res.optimal_solution_count == 1


def test_unreachable_target_is_infeasible():
    preds = [P(1, (0, 0, 10, 10), 0.9)]
    gts = [G(1, (0, 0, 10, 10)), G(7, (100, 100, 110, 110))]
    res = solve(preds, gts, [7], 0.5, [])
    assert res.status == "infeasible"
    assert "7" in res.explanation
    assert res.all_exempted.missed_gt_ids == [7]


def test_empty_predictions_infeasible_when_target_exists():
    res = solve([], [G(1, (0, 0, 1, 1))], [1], 0.5, [])
    assert res.status == "infeasible"
    assert res.baseline.kept_ids == []


def test_ineffective_pair_is_ignored():
    # 两框 IoU=0，不超过阈值：豁免无意义，应列入 ignored
    preds = [P(1, (0, 0, 10, 10), 0.9), P(2, (100, 100, 110, 110), 0.8)]
    gts = [G(1, (0, 0, 10, 10)), G(2, (100, 100, 110, 110))]
    res = solve(preds, gts, [1, 2], 0.5, [(1, 2)])
    assert res.status == "solved"
    assert res.chosen_pairs == []
    assert res.ineffective_pairs == [(1, 2)]


def test_is_feasible_fp_guard():
    res = EvalResult(kept_ids=[1, 2], suppressed=[], matches=[(1, 10, 0.9)],
                     false_positive_ids=[2], hit_gt_ids=[10], missed_gt_ids=[])
    assert is_feasible(res, required_hits={10}, max_fp=1)
    assert not is_feasible(res, required_hits={10}, max_fp=0)   # 误报增加 → 不可行
    assert not is_feasible(res, required_hits={10, 11}, max_fp=1)  # 命中缺失 → 不可行
