"""漏检复核核心算法。

规则（与需求约定一致）：
1. 预测框按置信度降序（并列取编号较小者）做贪心 NMS：
   只有被保留的框能抑制 IoU **严格大于** 阈值的后续框。
2. 按同一顺序把每个保留框配给 IoU 最大且 **至少 0.5** 的未配真值；
   IoU 并列时取编号较小的真值。未配上的保留框计为误报。
3. 豁免一对预测框 = 仅解除这两框之间的相互抑制（不影响其他框对）。
4. 目标：用最少豁免对找回全部指定漏检真值，同时
   - 无豁免时原有的命中（按真值编号计）必须全部保留；
   - 误报数不得超过无豁免时。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

MATCH_IOU_FLOOR = 0.5  # 保留框匹配真值的最小 IoU（含）


@dataclass(frozen=True)
class Prediction:
    id: int
    box: tuple[float, float, float, float]  # [x1, y1, x2, y2]
    score: float


@dataclass(frozen=True)
class GroundTruth:
    id: int
    box: tuple[float, float, float, float]


@dataclass
class EvalResult:
    """一次「NMS + 匹配」的结果。"""

    kept_ids: list[int]                        # 保留的预测框编号（按处理顺序）
    suppressed: list[tuple[int, int, float]]   # (被抑制框, 抑制者, 相互 IoU)
    matches: list[tuple[int, int, float]]      # (预测框, 真值, IoU)
    false_positive_ids: list[int]              # 未配上的保留框
    hit_gt_ids: list[int]                      # 命中的真值编号（升序）
    missed_gt_ids: list[int]                   # 未命中的真值编号（升序）


@dataclass
class CandidateState:
    """某个真值的一个候选预测框在某一版本重放中的状态。"""

    prediction_id: int
    iou: float                               # 该框与目标的 IoU（≥ MATCH_IOU_FLOOR）
    nms_status: str                          # "kept" | "suppressed"
    suppressed_by_prediction_id: int | None  # NMS 抑制者（保留框）
    suppression_iou: float | None            # 与抑制者的 IoU
    matched_gt_id: int | None                # 保留框最终配上的真值（可为其他真值）


@dataclass
class TargetVersionState:
    """指定真值在某一版本重放中的命中状态与候选框明细。"""

    status: str                              # "hit" | "missed"
    matched_prediction_id: int | None        # 命中该真值的保留框
    candidates: list[CandidateState]         # IoU ≥ 0.5 的候选框，按 IoU 降序、编号升序


@dataclass
class CauseInfo:
    """一次「漏检」的原因归因。"""

    type: str                                # "nms_suppressed" | "gt_competition" | "no_coverage"
    version: str                             # "baseline" | "upgraded"：原因取自哪一版的重放
    prediction_ids: list[int]                # 与原因对应的预测框编号
    suppressed_by_prediction_ids: list[int]  # NMS 原因：抑制这些候选框的保留框编号
    competing_gt_ids: list[int]              # 竞争原因：保留框实际配上的其他真值编号
    nearest_prediction_id: int | None        # 全局 IoU 最大的预测框（含低于匹配下限的）
    nearest_iou: float | None
    description: str


@dataclass
class TargetAttribution:
    """单个目标真值在两版重放之间的状态与变化原因。"""

    gt_id: int
    baseline: TargetVersionState
    upgraded: TargetVersionState
    changed: bool
    change: str                              # "unchanged" | "hit_to_missed" | "missed_to_hit"
    cause: CauseInfo | None


@dataclass
class AttributionResult:
    targets: list[TargetAttribution]
    baseline: EvalResult
    upgraded: EvalResult


@dataclass
class SolveResult:
    status: str                                # "solved" | "infeasible"
    baseline: EvalResult                       # 无豁免时的结果
    after: EvalResult | None                   # 采用最少豁免后的结果（无解时为 None）
    all_exempted: EvalResult | None            # 豁免全部有效候选对的结果（无解诊断用）
    chosen_pairs: list[tuple[int, int]]        # 选中的豁免对（预测框编号）
    optimal_solution_count: int | None         # 达到最少豁免数的可行方案个数
    ineffective_pairs: list[tuple[int, int]]   # 相互 IoU 未超阈值、豁免无意义的候选对
    explanation: str


def iou(a: tuple[float, float, float, float],
        b: tuple[float, float, float, float]) -> float:
    """两个 [x1, y1, x2, y2] 框的交并比。"""
    ix1 = a[0] if a[0] > b[0] else b[0]
    iy1 = a[1] if a[1] > b[1] else b[1]
    ix2 = a[2] if a[2] < b[2] else b[2]
    iy2 = a[3] if a[3] < b[3] else b[3]
    iw = ix2 - ix1
    ih = iy2 - iy1
    inter = iw * ih if (iw > 0.0 and ih > 0.0) else 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _iou_matrices(
    preds: list[Prediction], gts: list[GroundTruth]
) -> tuple[list[list[float]], list[list[float]]]:
    """预计算预测框两两 IoU 与预测框-真值 IoU 矩阵。"""
    n, m = len(preds), len(gts)
    pp_iou = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            v = iou(preds[i].box, preds[j].box)
            pp_iou[i][j] = v
            pp_iou[j][i] = v
    pg_iou = [[iou(preds[i].box, gts[g].box) for g in range(m)] for i in range(n)]
    return pp_iou, pg_iou


def evaluate(
    preds: list[Prediction],
    gts: list[GroundTruth],
    iou_threshold: float,
    exempt_pairs: frozenset[frozenset[int]],
    pp_iou: list[list[float]],
    pg_iou: list[list[float]],
) -> EvalResult:
    """按给定豁免对集合执行 NMS 与匹配。

    exempt_pairs 中每个元素为两个预测框编号组成的 frozenset。
    pp_iou[i][j] 为预测框 i、j 的 IoU；pg_iou[i][g] 为预测框 i 与真值 g 的 IoU。
    """
    # 置信度降序，并列取编号较小者
    order = sorted(range(len(preds)), key=lambda i: (-preds[i].score, preds[i].id))

    kept: list[int] = []
    suppressed: list[tuple[int, int, float]] = []
    for i in order:
        blocker: tuple[int, float] | None = None
        for k in kept:  # 只有保留框能抑制
            if frozenset((preds[k].id, preds[i].id)) in exempt_pairs:
                continue  # 该对框之间的抑制被豁免
            ov = pp_iou[k][i]
            if ov > iou_threshold:  # 严格大于阈值才抑制
                blocker = (k, ov)
                break
        if blocker is None:
            kept.append(i)
        else:
            suppressed.append((preds[i].id, preds[blocker[0]].id, blocker[1]))

    matched_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    fps: list[int] = []
    for i in kept:  # 按同一（置信度降序）顺序贪心匹配
        best = -1
        best_ov = MATCH_IOU_FLOOR
        for g in range(len(gts)):
            if gts[g].id in matched_gt:
                continue
            ov = pg_iou[i][g]
            if ov < MATCH_IOU_FLOOR:
                continue
            if best < 0 or ov > best_ov or (ov == best_ov and gts[g].id < gts[best].id):
                best = g
                best_ov = ov
        if best < 0:
            fps.append(preds[i].id)
        else:
            matched_gt.add(gts[best].id)
            matches.append((preds[i].id, gts[best].id, best_ov))

    all_gt_ids = {g.id for g in gts}
    return EvalResult(
        kept_ids=[preds[i].id for i in kept],
        suppressed=suppressed,
        matches=matches,
        false_positive_ids=fps,
        hit_gt_ids=sorted(matched_gt),
        missed_gt_ids=sorted(all_gt_ids - matched_gt),
    )


def is_feasible(res: EvalResult, required_hits: set[int], max_fp: int) -> bool:
    """可行性：指定+原有命中全部保留，且误报数不超过基线。"""
    return required_hits <= set(res.hit_gt_ids) and len(res.false_positive_ids) <= max_fp


def _version_state(
    res: EvalResult,
    gt_id: int,
    preds: list[Prediction],
    pg_iou: list[list[float]],
    gt_index: dict[int, int],
) -> TargetVersionState:
    """从一次重放结果中提取指定真值的命中状态与候选框明细。"""
    g = gt_index[gt_id]
    matched_by_pid = {pid: gid for pid, gid, _ in res.matches}
    suppressed_by = {pid: (by, v) for pid, by, v in res.suppressed}
    matched_pid = next((pid for pid, gid in matched_by_pid.items() if gid == gt_id), None)

    candidates: list[CandidateState] = []
    for i, pred in enumerate(preds):
        ov = pg_iou[i][g]
        if ov < MATCH_IOU_FLOOR:  # 与匹配规则一致：IoU ≥ 0.5 才算该真值的候选
            continue
        if pred.id in suppressed_by:
            by, by_iou = suppressed_by[pred.id]
            candidates.append(CandidateState(
                prediction_id=pred.id, iou=ov, nms_status="suppressed",
                suppressed_by_prediction_id=by, suppression_iou=by_iou,
                matched_gt_id=matched_by_pid.get(pred.id),
            ))
        else:
            candidates.append(CandidateState(
                prediction_id=pred.id, iou=ov, nms_status="kept",
                suppressed_by_prediction_id=None, suppression_iou=None,
                matched_gt_id=matched_by_pid.get(pred.id),
            ))
    candidates.sort(key=lambda c: (-c.iou, c.prediction_id))

    return TargetVersionState(
        status="hit" if matched_pid is not None else "missed",
        matched_prediction_id=matched_pid,
        candidates=candidates,
    )


def _nearest_prediction(
    preds: list[Prediction], pg_iou: list[list[float]], g: int
) -> tuple[int | None, float | None]:
    """全局 IoU 最大的预测框（含低于匹配下限者）；并列取编号较小者。"""
    if not preds:
        return None, None
    best = min(range(len(preds)), key=lambda i: (-pg_iou[i][g], preds[i].id))
    return preds[best].id, pg_iou[best][g]


def _cause(
    gt_id: int,
    state: TargetVersionState,
    version: str,
    preds: list[Prediction],
    pg_iou: list[list[float]],
    gt_index: dict[int, int],
) -> CauseInfo:
    """按候选框状态区分三类漏检原因：NMS 抑制 / 与其他真值竞争 / 无覆盖。"""
    nearest_pid, nearest_ov = _nearest_prediction(preds, pg_iou, gt_index[gt_id])
    version_name = "基线版" if version == "baseline" else "升级版"

    kept = [c for c in state.candidates if c.nms_status == "kept"]
    if not kept:
        if state.candidates:
            sup_ids = [c.prediction_id for c in state.candidates]
            by_ids = sorted({c.suppressed_by_prediction_id for c in state.candidates})
            return CauseInfo(
                type="nms_suppressed",
                version=version,
                prediction_ids=sup_ids,
                suppressed_by_prediction_ids=by_ids,
                competing_gt_ids=[],
                nearest_prediction_id=nearest_pid,
                nearest_iou=nearest_ov,
                description=(
                    f"{version_name}中真值 {gt_id} 的候选框 {sup_ids} 均被 NMS 抑制"
                    f"（抑制者 {by_ids}），没有保留框可再参与匹配。"
                ),
            )
        return CauseInfo(
            type="no_coverage",
            version=version,
            prediction_ids=[],
            suppressed_by_prediction_ids=[],
            competing_gt_ids=[],
            nearest_prediction_id=nearest_pid,
            nearest_iou=nearest_ov,
            description=(
                f"{version_name}没有任何预测框与真值 {gt_id} 的 IoU 达到匹配下限 "
                f"{MATCH_IOU_FLOOR}"
                + (f"（最大 IoU {nearest_ov:.6f}，来自框 {nearest_pid}）。"
                   if nearest_ov is not None else "（没有预测框）。")
            ),
        )

    # 有保留候选框却仍未命中：这些框在贪心匹配中已被其他真值占用
    kept_pids = [c.prediction_id for c in kept]
    competing = sorted({c.matched_gt_id for c in kept if c.matched_gt_id is not None})
    if competing:
        competition_text = f"真值 {competing} 占用"
    else:
        competition_text = "其他真值占用或其余框未达匹配下限"
    return CauseInfo(
        type="gt_competition",
        version=version,
        prediction_ids=kept_pids,
        suppressed_by_prediction_ids=[],
        competing_gt_ids=competing,
        nearest_prediction_id=nearest_pid,
        nearest_iou=nearest_ov,
        description=(
            f"{version_name}中真值 {gt_id} 的保留候选框 {kept_pids} "
            f"在贪心匹配中先被{competition_text}，未轮到该真值。"
        ),
    )


def attribute(
    gts: list[GroundTruth],
    target_gt_ids: list[int],
    iou_threshold: float,
    baseline_preds: list[Prediction],
    upgraded_preds: list[Prediction],
) -> AttributionResult:
    """在两版抑制前预测框上重放同一套 NMS+匹配，逐目标归因状态变化。

    两版共享同一份真值、NMS 阈值与编号规则；各版本独立重放，互不影响。
    命中 → 漏检：原因取自升级版重放（新出现的漏检为何发生）；
    漏检 → 命中：原因取自基线版重放（原先的漏检源自何处，参照 /review 的 MVP 思路）。
    """
    pp_base, pg_base = _iou_matrices(baseline_preds, gts)
    pp_up, pg_up = _iou_matrices(upgraded_preds, gts)
    base_res = evaluate(baseline_preds, gts, iou_threshold, frozenset(), pp_base, pg_base)
    up_res = evaluate(upgraded_preds, gts, iou_threshold, frozenset(), pp_up, pg_up)
    gt_index = {g.id: i for i, g in enumerate(gts)}

    targets: list[TargetAttribution] = []
    for gt_id in sorted({int(t) for t in target_gt_ids}):
        base_state = _version_state(base_res, gt_id, baseline_preds, pg_base, gt_index)
        up_state = _version_state(up_res, gt_id, upgraded_preds, pg_up, gt_index)
        cause: CauseInfo | None = None
        if base_state.status == up_state.status:
            changed, change = False, "unchanged"
        elif base_state.status == "hit":
            changed, change = True, "hit_to_missed"
            cause = _cause(gt_id, up_state, "upgraded", upgraded_preds, pg_up, gt_index)
        else:
            changed, change = True, "missed_to_hit"
            cause = _cause(
                gt_id, base_state, "baseline", baseline_preds, pg_base, gt_index)
        targets.append(TargetAttribution(
            gt_id=gt_id,
            baseline=base_state,
            upgraded=up_state,
            changed=changed,
            change=change,
            cause=cause,
        ))

    return AttributionResult(targets=targets, baseline=base_res, upgraded=up_res)


def solve(
    preds: list[Prediction],
    gts: list[GroundTruth],
    target_gt_ids: list[int],
    iou_threshold: float,
    candidate_pairs: list[tuple[int, int]],
) -> SolveResult:
    """搜索找回指定漏检所需的最少豁免框对。"""
    targets = {int(t) for t in target_gt_ids}

    # 预计算 IoU 矩阵，避免在子集枚举中重复计算
    n, m = len(preds), len(gts)
    pp_iou, pg_iou = _iou_matrices(preds, gts)

    idx_of = {p.id: i for i, p in enumerate(preds)}

    # 候选对去重并规范化（小编号在前）；相互 IoU 未超阈值的对永远不会产生抑制，豁免无意义
    normalized = sorted({(min(a, b), max(a, b)) for a, b in candidate_pairs})
    effective: list[tuple[int, int]] = []
    ineffective: list[tuple[int, int]] = []
    for pair in normalized:
        a, b = pair
        if pp_iou[idx_of[a]][idx_of[b]] > iou_threshold:
            effective.append(pair)
        else:
            ineffective.append(pair)

    baseline = evaluate(preds, gts, iou_threshold, frozenset(), pp_iou, pg_iou)
    base_hits = set(baseline.hit_gt_ids)
    base_fp = len(baseline.false_positive_ids)
    required = base_hits | targets  # 原有命中 + 指定漏检都必须命中

    k = len(effective)
    for size in range(0, k + 1):
        first_found: tuple[tuple[int, ...], EvalResult] | None = None
        feasible_count = 0
        for combo in combinations(range(k), size):
            exempt = frozenset(frozenset(effective[i]) for i in combo)
            res = evaluate(preds, gts, iou_threshold, exempt, pp_iou, pg_iou)
            if is_feasible(res, required, base_fp):
                feasible_count += 1
                if first_found is None:
                    first_found = (combo, res)
        if first_found is not None:
            combo, res = first_found
            chosen = [effective[i] for i in combo]
            if size == 0:
                explanation = "无需豁免：无豁免时全部指定真值已命中，原有命中与误报数自然满足约束。"
            else:
                pairs_str = "、".join(f"({a},{b})" for a, b in chosen)
                explanation = (
                    f"最少需豁免 {size} 对预测框：{pairs_str}。"
                    f"豁免后指定漏检全部找回，原有命中全部保留，"
                    f"误报数 {base_fp} → {len(res.false_positive_ids)}（未增加）。"
                )
                if feasible_count > 1:
                    explanation += f"（另有 {feasible_count - 1} 个同为最少的可行方案，返回按候选对顺序的第一个。）"
            return SolveResult(
                status="solved",
                baseline=baseline,
                after=res,
                all_exempted=None,
                chosen_pairs=chosen,
                optimal_solution_count=feasible_count,
                ineffective_pairs=ineffective,
                explanation=explanation,
            )

    # 无解：用「豁免全部有效候选对」做诊断说明
    exempt_all = frozenset(frozenset(p) for p in effective)
    res_all = evaluate(preds, gts, iou_threshold, exempt_all, pp_iou, pg_iou)
    hits_all = set(res_all.hit_gt_ids)
    problems: list[str] = []
    unrecovered = sorted(targets - hits_all)
    if unrecovered:
        problems.append(f"指定真值 {unrecovered} 在豁免全部 {k} 对有效候选框后仍未命中")
    lost = sorted(base_hits - hits_all)
    if lost:
        problems.append(f"原有命中 {lost} 无法保留")
    fp_all = len(res_all.false_positive_ids)
    if fp_all > base_fp:
        problems.append(f"误报数将由 {base_fp} 增至 {fp_all}")
    if not problems:
        problems.append("约束在部分子集上互相冲突，无法同时满足")
    explanation = (
        "无解：候选豁免对的任何子集都无法同时满足「找回指定漏检、原有命中保留、误报不增」。"
        + "；".join(problems)
        + "。"
    )
    if k == 0:
        explanation += "（没有可改变结果的有效候选对：候选对相互 IoU 均未超过抑制阈值，或未提供候选对。）"
    return SolveResult(
        status="infeasible",
        baseline=baseline,
        after=None,
        all_exempted=res_all,
        chosen_pairs=[],
        optimal_solution_count=None,
        ineffective_pairs=ineffective,
        explanation=explanation,
    )
