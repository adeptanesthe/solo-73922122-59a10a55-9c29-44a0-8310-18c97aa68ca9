"""请求 / 响应数据模型与入参校验。"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_BOXES = 60          # /review：预测框、真值框各至多 60 个
MAX_EXEMPT_PAIRS = 10   # /review：可豁免候选框对至多 10 对
MAX_BOXES_ATTRIBUTION = 120  # /review/attribution：每版预测框、真值框各至多 120 个


def _check_finite(coords: list[float]) -> list[float]:
    if any(not math.isfinite(c) for c in coords):
        raise ValueError("坐标/分数必须是有限数值")
    return coords


class PredictionIn(BaseModel):
    id: int = Field(ge=0, description="预测框编号（同一张图内唯一）")
    box: list[float] = Field(min_length=4, max_length=4,
                             description="[x1, y1, x2, y2]，要求 x2>x1 且 y2>y1")
    score: float = Field(ge=0.0, le=1.0, description="置信度，取值 [0, 1]")

    @field_validator("box")
    @classmethod
    def _box_valid(cls, v: list[float]) -> list[float]:
        _check_finite(v)
        if not (v[2] > v[0] and v[3] > v[1]):
            raise ValueError("box 必须满足 x2>x1 且 y2>y1（格式 [x1, y1, x2, y2]）")
        return v

    @field_validator("score")
    @classmethod
    def _score_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("score 必须是有限数值")
        return v


class GroundTruthIn(BaseModel):
    id: int = Field(ge=0, description="真值框编号（同一张图内唯一）")
    box: list[float] = Field(min_length=4, max_length=4,
                             description="[x1, y1, x2, y2]，要求 x2>x1 且 y2>y1")

    @field_validator("box")
    @classmethod
    def _box_valid(cls, v: list[float]) -> list[float]:
        _check_finite(v)
        if not (v[2] > v[0] and v[3] > v[1]):
            raise ValueError("box 必须满足 x2>x1 且 y2>y1（格式 [x1, y1, x2, y2]）")
        return v


class ReviewRequest(BaseModel):
    predictions: list[PredictionIn] = Field(
        max_length=MAX_BOXES, description="NMS 抑制前的预测框，至多 60 个")
    ground_truths: list[GroundTruthIn] = Field(
        max_length=MAX_BOXES, description="真值框，至多 60 个")
    target_gt_ids: list[int] = Field(
        default_factory=list, max_length=MAX_BOXES,
        description="待找回（必须命中）的真值编号")
    iou_threshold: float = Field(ge=0.0, le=1.0,
                                 description="NMS 抑制阈值：IoU 严格大于它才抑制")
    exemptible_pairs: list[list[int]] = Field(
        default_factory=list, max_length=MAX_EXEMPT_PAIRS,
        description="可豁免的预测框编号对，至多 10 对；每对格式 [idA, idB]")

    @field_validator("exemptible_pairs")
    @classmethod
    def _pairs_valid(cls, v: list[list[int]]) -> list[list[int]]:
        for pair in v:
            if len(pair) != 2:
                raise ValueError("每个豁免对必须恰好包含 2 个预测框编号")
            if pair[0] == pair[1]:
                raise ValueError("豁免对的两个编号不能相同")
        return v

    @model_validator(mode="after")
    def _cross_check(self) -> "ReviewRequest":
        pred_ids = [p.id for p in self.predictions]
        if len(set(pred_ids)) != len(pred_ids):
            raise ValueError("预测框编号存在重复")
        gt_ids = [g.id for g in self.ground_truths]
        if len(set(gt_ids)) != len(gt_ids):
            raise ValueError("真值框编号存在重复")
        pred_id_set, gt_id_set = set(pred_ids), set(gt_ids)
        unknown_targets = sorted(set(self.target_gt_ids) - gt_id_set)
        if unknown_targets:
            raise ValueError(f"target_gt_ids 包含不存在的真值编号: {unknown_targets}")
        for pair in self.exemptible_pairs:
            unknown = [i for i in pair if i not in pred_id_set]
            if unknown:
                raise ValueError(f"豁免对 {pair} 包含不存在的预测框编号: {unknown}")
        return self


class SuppressionOut(BaseModel):
    prediction_id: int
    suppressed_by: int
    iou: float


class MatchOut(BaseModel):
    prediction_id: int
    gt_id: int
    iou: float


class MatchReportOut(BaseModel):
    kept_prediction_ids: list[int] = Field(description="NMS 后保留的预测框（按置信度降序）")
    suppressed: list[SuppressionOut] = Field(description="被抑制的预测框及其抑制者")
    matches: list[MatchOut] = Field(description="保留框与真值的配对明细")
    false_positive_prediction_ids: list[int] = Field(description="未配上真值的保留框（误报）")
    false_positive_count: int
    hit_gt_ids: list[int] = Field(description="被命中的真值编号")
    missed_gt_ids: list[int] = Field(description="未被命中的真值编号")
    recovered_target_gt_ids: list[int] = Field(description="指定待找回真值中已命中的部分")
    unrecovered_target_gt_ids: list[int] = Field(description="指定待找回真值中仍未命中的部分")


class ReviewResponse(BaseModel):
    status: str = Field(description="solved=找到最少豁免方案；infeasible=无解")
    min_exemption_count: int | None = Field(description="所需最少豁免对数（无解时为 null）")
    chosen_exempt_pairs: list[list[int]] = Field(description="选中的豁免框对")
    optimal_solution_count: int | None = Field(
        description="达到最少豁免数的可行方案个数（无解时为 null）")
    ignored_ineffective_pairs: list[list[int]] = Field(
        description="相互 IoU 未超过抑制阈值、豁免无意义而被忽略的候选对")
    explanation: str = Field(description="结论说明；无解时说明原因")
    baseline: MatchReportOut = Field(description="无豁免时的 NMS+匹配明细")
    after: MatchReportOut | None = Field(description="采用最少豁免后的明细（无解时为 null）")
    all_exempted: MatchReportOut | None = Field(
        description="豁免全部有效候选对后的明细（仅无解时给出，用于诊断）")


# ---------------- 版本归因 ----------------

class AttributionRequest(BaseModel):
    ground_truths: list[GroundTruthIn] = Field(
        max_length=MAX_BOXES_ATTRIBUTION,
        description="两版共享的真值框，至多 120 个")
    iou_threshold: float = Field(ge=0.0, le=1.0,
                                 description="NMS 抑制阈值：IoU 严格大于它才抑制")
    target_gt_ids: list[int] = Field(
        min_length=1, max_length=MAX_BOXES_ATTRIBUTION,
        description="需要逐目标归因的真值编号，至少 1 个")
    baseline_predictions: list[PredictionIn] = Field(
        max_length=MAX_BOXES_ATTRIBUTION,
        description="基线版 NMS 抑制前的预测框，至多 120 个")
    upgraded_predictions: list[PredictionIn] = Field(
        max_length=MAX_BOXES_ATTRIBUTION,
        description="升级版 NMS 抑制前的预测框，至多 120 个")

    @model_validator(mode="after")
    def _cross_check(self) -> "AttributionRequest":
        gt_ids = [g.id for g in self.ground_truths]
        if len(set(gt_ids)) != len(gt_ids):
            raise ValueError("真值框编号存在重复")
        for name in ("baseline_predictions", "upgraded_predictions"):
            preds = getattr(self, name)
            pred_ids = [p.id for p in preds]
            if len(set(pred_ids)) != len(pred_ids):
                raise ValueError(f"{name} 的预测框编号存在重复")
        gt_id_set = set(gt_ids)
        unknown = sorted(set(self.target_gt_ids) - gt_id_set)
        if unknown:
            raise ValueError(f"target_gt_ids 包含不存在的真值编号: {unknown}")
        return self


class CandidateStateOut(BaseModel):
    prediction_id: int = Field(description="候选预测框编号")
    iou: float = Field(description="该框与目标真值的 IoU（≥ 0.5 才会列为候选）")
    nms_status: Literal["kept", "suppressed"] = Field(
        description="NMS 后状态：kept=保留，suppressed=被抑制")
    suppressed_by_prediction_id: int | None = Field(
        description="NMS 抑制者编号；未被抑制时为 null")
    suppression_iou: float | None = Field(description="与抑制者的 IoU；未被抑制时为 null")
    matched_gt_id: int | None = Field(
        description=(
            "最终配上的真值编号（可能不是本目标）；被抑制或本框未配上任何真值时为 null"))


class TargetVersionStateOut(BaseModel):
    status: Literal["hit", "missed"] = Field(description="该真值在本版本中是否命中")
    matched_prediction_id: int | None = Field(
        description="命中该真值的保留框编号；漏检时为 null")
    candidates: list[CandidateStateOut] = Field(
        description="本版本中该真值的候选框（IoU ≥ 0.5），按 IoU 降序、编号升序")


class CauseOut(BaseModel):
    type: Literal["nms_suppressed", "gt_competition", "no_coverage"] = Field(
        description=(
            "漏检原因：nms_suppressed=候选框全部被 NMS 抑制；"
            "gt_competition=保留框在贪心匹配中与其他真值竞争失败；"
            "no_coverage=没有预测框与真值的 IoU 达到 0.5"))
    version: Literal["baseline", "upgraded"] = Field(
        description=(
            "原因取自哪一版的重放：hit_to_missed 取 upgraded（新漏检为何发生），"
            "missed_to_hit 取 baseline（原漏检源自何处）"))
    prediction_ids: list[int] = Field(
        description="与该原因对应的候选预测框编号（竞争原因下为保留候选框）")
    suppressed_by_prediction_ids: list[int] = Field(
        description="NMS 原因下抑制这些候选框的保留框编号")
    competing_gt_ids: list[int] = Field(
        description="竞争原因下保留候选框实际配上的其他真值编号")
    nearest_prediction_id: int | None = Field(
        description="与真值 IoU 最大的预测框编号（含 IoU 低于 0.5 者）；无预测框时为 null")
    nearest_iou: float | None = Field(description="上述最大 IoU；无预测框时为 null")
    description: str = Field(description="中文说明")


class TargetAttributionOut(BaseModel):
    gt_id: int = Field(description="目标真值编号")
    changed: bool = Field(description="两版命中状态是否发生变化")
    change: Literal["unchanged", "hit_to_missed", "missed_to_hit"] = Field(
        description="状态变化方向")
    baseline: TargetVersionStateOut = Field(description="基线版重放明细")
    upgraded: TargetVersionStateOut = Field(description="升级版重放明细")
    cause: CauseOut | None = Field(
        description=(
            "状态变化时的原因归因（含变化方向对应的版本与框号）；"
            "状态未变时为 null"))


class AttributionResponse(BaseModel):
    targets: list[TargetAttributionOut] = Field(
        description="按真值编号升序的逐目标归因结果")
    changed_target_count: int = Field(description="状态发生变化的目标数量")
    hit_to_missed_gt_ids: list[int] = Field(description="由命中变为漏检的真值编号")
    missed_to_hit_gt_ids: list[int] = Field(description="由漏检变为命中的真值编号")
    baseline: MatchReportOut = Field(description="基线版整体 NMS+匹配明细")
    upgraded: MatchReportOut = Field(description="升级版整体 NMS+匹配明细")
