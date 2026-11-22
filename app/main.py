"""FastAPI 入口：釉面缺陷漏检复核服务。"""
from __future__ import annotations

from fastapi import FastAPI

from . import solver
from .models import (
    MatchOut,
    MatchReportOut,
    ReviewRequest,
    ReviewResponse,
    SuppressionOut,
)

app = FastAPI(
    title="釉面缺陷漏检复核 API",
    version="1.0.0",
    description=(
        "针对相邻釉面缺陷在 NMS 后只剩一个框的漏检问题："
        "在抑制前预测框上重放贪心 NMS 与匹配，"
        "搜索找回指定漏检真值所需的最少豁免框对。"
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
