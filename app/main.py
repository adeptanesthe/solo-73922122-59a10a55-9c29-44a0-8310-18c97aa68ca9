"""FastAPI 入口：釉面缺陷漏检复核服务。"""
from __future__ import annotations

from fastapi import FastAPI

from . import solver
from .models import (
    AttributionRequest,
    AttributionResponse,
    CandidateStateOut,
    CauseOut,
    MatchOut,
    MatchReportOut,
    ReviewRequest,
    ReviewResponse,
    SuppressionOut,
    TargetAttributionOut,
    TargetVersionStateOut,
)

app = FastAPI(
    title="釉面缺陷漏检复核 API",
    version="1.1.0",
    description=(
        "针对相邻釉面缺陷在 NMS 后只剩一个框的漏检问题："
        "在抑制前预测框上重放贪心 NMS 与匹配，"
        "搜索找回指定漏检真值所需的最少豁免框对；"
        "并可按版本归因，区分漏检源自 NMS 抑制、真值竞争或无覆盖。"
    ),
)


def _report(res: solver.EvalResult, targets: set[int]) -> MatchReportOut:
    hits = set(res.hit_gt_ids)
    return MatchReportOut(
        kept_prediction_ids=res.kept_ids,
        suppressed=[
            SuppressionOut(prediction_id=pid, suppressed_by=by, iou=round(v, 6))
            for pid, by, v in res.suppressed
        ],
        matches=[
            MatchOut(prediction_id=pid, gt_id=gid, iou=round(v, 6))
            for pid, gid, v in res.matches
        ],
        false_positive_prediction_ids=res.false_positive_ids,
        false_positive_count=len(res.false_positive_ids),
        hit_gt_ids=res.hit_gt_ids,
        missed_gt_ids=res.missed_gt_ids,
        recovered_target_gt_ids=sorted(targets & hits),
        unrecovered_target_gt_ids=sorted(targets - hits),
    )


@app.get("/health", summary="健康检查")
def health() -> dict:
    return {"status": "ok"}


@app.post(
    "/api/v1/review",
    response_model=ReviewResponse,
    summary="漏检复核：搜索找回指定漏检的最少豁免框对",
)
def review(req: ReviewRequest) -> ReviewResponse:
    preds = [
        solver.Prediction(id=p.id, box=tuple(p.box), score=p.score)
        for p in req.predictions
    ]
    gts = [solver.GroundTruth(id=g.id, box=tuple(g.box)) for g in req.ground_truths]
    pairs = [tuple(pair) for pair in req.exemptible_pairs]

    result = solver.solve(
        preds=preds,
        gts=gts,
        target_gt_ids=req.target_gt_ids,
        iou_threshold=req.iou_threshold,
        candidate_pairs=pairs,
    )

    targets = set(req.target_gt_ids)
    return ReviewResponse(
        status=result.status,
        min_exemption_count=(
            len(result.chosen_pairs) if result.status == "solved" else None
        ),
        chosen_exempt_pairs=[list(p) for p in result.chosen_pairs],
        optimal_solution_count=result.optimal_solution_count,
        ignored_ineffective_pairs=[list(p) for p in result.ineffective_pairs],
        explanation=result.explanation,
        baseline=_report(result.baseline, targets),
        after=_report(result.after, targets) if result.after is not None else None,
        all_exempted=(
            _report(result.all_exempted, targets)
            if result.all_exempted is not None
            else None
        ),
    )


def _version_state(state: solver.TargetVersionState) -> TargetVersionStateOut:
    return TargetVersionStateOut(
        status=state.status,
        matched_prediction_id=state.matched_prediction_id,
        candidates=[
            CandidateStateOut(
                prediction_id=c.prediction_id,
                iou=round(c.iou, 6),
                nms_status=c.nms_status,
                suppressed_by_prediction_id=c.suppressed_by_prediction_id,
                suppression_iou=(
                    round(c.suppression_iou, 6)
                    if c.suppression_iou is not None else None
                ),
                matched_gt_id=c.matched_gt_id,
            )
            for c in state.candidates
        ],
    )


@app.post(
    "/api/v1/review/attribution",
    response_model=AttributionResponse,
    summary="版本归因：重放两版 NMS+匹配，逐目标区分漏检原因",
)
def attribution(req: AttributionRequest) -> AttributionResponse:
    gts = [solver.GroundTruth(id=g.id, box=tuple(g.box)) for g in req.ground_truths]

    def to_preds(items):
        return [
            solver.Prediction(id=p.id, box=tuple(p.box), score=p.score) for p in items
        ]

    result = solver.attribute(
        gts=gts,
        target_gt_ids=req.target_gt_ids,
        iou_threshold=req.iou_threshold,
        baseline_preds=to_preds(req.baseline_predictions),
        upgraded_preds=to_preds(req.upgraded_predictions),
    )

    targets = set(req.target_gt_ids)
    hit_to_missed = sorted(t.gt_id for t in result.targets if t.change == "hit_to_missed")
    missed_to_hit = sorted(t.gt_id for t in result.targets if t.change == "missed_to_hit")

    return AttributionResponse(
        targets=[
            TargetAttributionOut(
                gt_id=t.gt_id,
                changed=t.changed,
                change=t.change,
                baseline=_version_state(t.baseline),
                upgraded=_version_state(t.upgraded),
                cause=(
                    CauseOut(
                        type=t.cause.type,
                        version=t.cause.version,
                        prediction_ids=t.cause.prediction_ids,
                        suppressed_by_prediction_ids=(
                            t.cause.suppressed_by_prediction_ids),
                        competing_gt_ids=t.cause.competing_gt_ids,
                        nearest_prediction_id=t.cause.nearest_prediction_id,
                        nearest_iou=(
                            round(t.cause.nearest_iou, 6)
                            if t.cause.nearest_iou is not None else None
                        ),
                        description=t.cause.description,
                    )
                    if t.cause is not None else None
                ),
            )
            for t in result.targets
        ],
        changed_target_count=len(hit_to_missed) + len(missed_to_hit),
        hit_to_missed_gt_ids=hit_to_missed,
        missed_to_hit_gt_ids=missed_to_hit,
        baseline=_report(result.baseline, targets),
        upgraded=_report(result.upgraded, targets),
    )
