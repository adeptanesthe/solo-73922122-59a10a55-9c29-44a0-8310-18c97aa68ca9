# 釉面缺陷漏检复核 API

相邻的釉面缺陷在 NMS 后可能只剩一个框，造成漏检。本服务在**抑制前**的预测框上
重放「贪心 NMS + 贪心匹配」，并搜索：为找回指定漏检真值，最少需要豁免多少对
预测框之间的抑制。

## 规则约定

1. **NMS**：预测框按置信度降序（分数并列取编号较小者）做贪心 NMS；只有被保留
   的框能抑制 IoU **严格大于** `iou_threshold` 的后续框。
2. **匹配**：按同一顺序，把每个保留框配给 IoU 最大且 **≥ 0.5** 的未配真值；
   IoU 并列取编号较小的真值。未配上的保留框计为**误报**。
3. **豁免**：一对豁免框仅解除这两个框之间的相互抑制，不影响其他框对；被释放
   的框之后同样可以抑制别人（可能产生级联）。
4. **目标**：找到**最少**的豁免对，使得
   - 指定的待找回真值全部命中；
   - 无豁免时原有命中的真值（按真值编号计，允许更换命中它的预测框）全部保留；
   - 误报数不超过无豁免时。
   同为最少的方案有多个时，返回按候选对顺序的第一个，并在
   `optimal_solution_count` 中给出方案总数。
5. 候选对中两框 IoU 未超过抑制阈值的，豁免无意义，会被列入
   `ignored_ineffective_pairs` 而不参与搜索。

## 限制

- `/review`：预测框、真值框各至多 **60** 个；可豁免候选框对至多 **10** 对。
- `/review/attribution`：两版预测框各至多 **120** 个，真值至多 **120** 个。
- 框格式 `[x1, y1, x2, y2]`，要求 `x2>x1`、`y2>y1`；置信度 ∈ [0, 1]；
  `iou_threshold` ∈ [0, 1]；编号在同一张图内唯一（非负整数）。

## 启动（Docker Compose）

```bash
docker compose up --build
# 服务监听 http://localhost:8000 ，交互式文档见 http://localhost:8000/docs
```

## 本地开发

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --reload
.venv/bin/pytest -q          # 运行测试
```

## API

### `GET /health`

健康检查，返回 `{"status": "ok"}`。

### `POST /api/v1/review`

请求体：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `predictions` | array | 抑制前预测框 `[{id, box, score}]`，≤60 |
| `ground_truths` | array | 真值框 `[{id, box}]`，≤60 |
| `target_gt_ids` | int[] | 待找回的真值编号 |
| `iou_threshold` | float | NMS 抑制阈值（IoU 严格大于它才抑制） |
| `exemptible_pairs` | int[][] | 可豁免的预测框编号对，≤10 对 |

响应要点：

- `status`: `solved` / `infeasible`
- `min_exemption_count` / `chosen_exempt_pairs`: 最少豁免对数与选中的框对
- `baseline`: 无豁免时的 NMS+匹配明细（保留框、被抑制框、配对、误报、漏检）
- `after`: 采用最少豁免后的同样明细（无解时为 `null`）
- `all_exempted`: 豁免全部有效候选对后的明细（仅无解时给出，用于诊断）
- `explanation`: 结论说明；无解时列出未满足的条件

### 可调用示例

启动服务后：

```bash
# 场景一：两个相邻缺陷，P2 命中 GT102 但被 P1 抑制
curl -sS -X POST http://localhost:8000/api/v1/review \
  -H 'Content-Type: application/json' \
  -d @examples/request_solved.json
```

关键响应（完整字段略）：

```json
{
  "status": "solved",
  "min_exemption_count": 1,
  "chosen_exempt_pairs": [[1, 2]],
  "explanation": "最少需豁免 1 对预测框：(1,2)。豁免后指定漏检全部找回，原有命中全部保留，误报数 1 → 1（未增加）。",
  "baseline":  { "kept_prediction_ids": [1, 3], "missed_gt_ids": [102], "false_positive_count": 1 },
  "after":     { "kept_prediction_ids": [1, 2, 3], "hit_gt_ids": [101, 102], "false_positive_count": 1 }
}
```

```bash
# 场景二：GT103 无任何预测框覆盖，豁免也找不回 → 无解并说明原因
curl -sS -X POST http://localhost:8000/api/v1/review \
  -H 'Content-Type: application/json' \
  -d @examples/request_infeasible.json
```

```json
{
  "status": "infeasible",
  "min_exemption_count": null,
  "explanation": "无解：候选豁免对的任何子集都无法同时满足「找回指定漏检、原有命中保留、误报不增」。指定真值 [103] 在豁免全部 1 对有效候选框后仍未命中。"
}
```

也可以直接运行 `examples/call_api.sh`（服务需已启动）：

```bash
./examples/call_api.sh                                  # 场景一
./examples/call_api.sh http://localhost:8000 examples/request_infeasible.json   # 场景二
```

### `POST /api/v1/review/attribution`

版本归因：同一张照片共享一份真值、一个 `iou_threshold` 和一套编号规则，分别对
**基线版**与**升级版**抑制前的预测框重放与 `/review` 完全相同的贪心 NMS + 匹配
（无豁免、阈值严格大于才抑制、匹配下限 0.5 含等号、分数并列取小编号），再逐目标
对比两版状态并给出漏检原因。

请求体：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `ground_truths` | array | 两版共享的真值框 `[{id, box}]`，≤120 |
| `iou_threshold` | float | 两版共享的 NMS 抑制阈值，∈ [0, 1] |
| `target_gt_ids` | int[] | 需要逐目标归因的真值编号，至少 1 个，≤120 |
| `baseline_predictions` | array | 基线版抑制前预测框 `[{id, box, score}]`，≤120 |
| `upgraded_predictions` | array | 升级版抑制前预测框 `[{id, box, score}]`，≤120 |

框格式、`score` 取值、编号唯一性等约束与 `/review` 相同，不合法请求返回 422。

响应对每个目标给出：

- `baseline` / `upgraded`: 该真值在两版中的 `status`（`hit` / `missed`）、
  命中它的 `matched_prediction_id`，以及所有 IoU ≥ 0.5 的 `candidates`。
  每个候选框给出 `prediction_id`、与该真值的 `iou`、`nms_status`
  （`kept` / `suppressed`）、`suppressed_by_prediction_id` 与 `suppression_iou`、
  以及保留框最终配上的 `matched_gt_id`（保留但配给别的真值时可见）。
- `change`: `unchanged` / `hit_to_missed` / `missed_to_hit`。
- `cause`（仅状态变化时非空）：`type` + `version`，
  - `nms_suppressed`：目标的所有候选框都被 NMS 抑制。给出被抑制的
    `prediction_ids` 和抑制它们的 `suppressed_by_prediction_ids`。
  - `gt_competition`：目标有保留候选框，但在贪心匹配中先被其他真值占用。
    给出保留候选框 `prediction_ids` 与其实际配上的 `competing_gt_ids`。
  - `no_coverage`：没有任何预测框与该真值 IoU 达 0.5。`prediction_ids` 为空。
  - `version`：`hit_to_missed` 取 `upgraded`（新漏检为何发生）；
    `missed_to_hit` 取 `baseline`（原漏检源自何处）。
  - 另附 `nearest_prediction_id` / `nearest_iou`：与该真值 IoU 最大的预测框
    （含低于 0.5 者），用于定位“最近但还不够”的框。
- 顶层 `baseline` / `upgraded`: 两版整体 NMS+匹配明细，字段与 `/review`
  的 `baseline` 相同；`changed_target_count`、`hit_to_missed_gt_ids`、
  `missed_to_hit_gt_ids` 汇总变化；`targets` 按真值编号升序且去重。

```bash
curl -sS -X POST http://localhost:8000/api/v1/review/attribution \
  -H 'Content-Type: application/json' \
  -d @examples/request_attribution.json
```

关键响应（完整字段略）：

```json
{
  "targets": [
    {
      "gt_id": 102,
      "changed": true,
      "change": "hit_to_missed",
      "baseline": {"status": "hit", "matched_prediction_id": 2,
                   "candidates": [{"prediction_id": 2, "nms_status": "kept", "matched_gt_id": 102}]},
      "upgraded": {"status": "missed", "matched_prediction_id": null,
                   "candidates": [{"prediction_id": 2, "nms_status": "suppressed",
                                   "suppressed_by_prediction_id": 9, "suppression_iou": 0.555556}]},
      "cause": {"type": "nms_suppressed", "version": "upgraded",
                "prediction_ids": [2], "suppressed_by_prediction_ids": [9]}
    }
  ]
}
```

## 实现说明

- 核心逻辑在 `app/solver.py`：先一次性预计算预测框两两 IoU 与预测框-真值 IoU
  矩阵，再对候选对的每个子集（≤2¹⁰）重放 NMS+匹配，按豁免对数从小到大找到
  第一个可行子集即为最优解；无解时用「全部豁免」的结果生成诊断说明。
- 版本归因复用同一套 `evaluate` 重放：两版各自独立计算 IoU 矩阵并重放，
  再从结果中提取逐目标候选状态。原因判定顺序为：无 IoU ≥ 0.5 候选 →
  `no_coverage`；有候选但全部被抑制 → `nms_suppressed`；有保留候选 →
  `gt_competition`。`/review` 的行为与响应未改动。
- 校验在 `app/models.py`（Pydantic）：数量上限、框合法性、编号唯一、
  目标真值与豁免对端点必须存在等，不合法请求返回 422 及中文错误信息。
- 测试在 `tests/`（59 个用例）：IoU 边界、严格大于阈值才抑制、只有保留框能
  抑制、分数/IoU 并列取小编号、匹配下限 0.5 含等号、级联需两对豁免、原有命中
  丢失判无解、误报不增约束、无效候选对忽略、版本归因三类原因与往返变化、
  API 端到端与全部 422 分支。
