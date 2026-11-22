"""请求 / 响应数据模型与入参校验。"""
from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_BOXES = 60          # 预测框、真值框各至多 60 个
MAX_EXEMPT_PAIRS = 10   # 可豁免候选框对至多 10 对


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
