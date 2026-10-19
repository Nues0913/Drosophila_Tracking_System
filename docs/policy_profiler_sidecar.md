# ROS policy profiler sidecar 格式

Policy profiler 預設關閉。Issue #52 新增的 trace 狀態欄位以 JSONL sidecar 的 `run_start.schema_version: 2` 標示。舊 sidecar 沒有 schema version；其中 `control_trace_status: "emitted"` 只表示 command 已交給 tracer，不代表執行過 `publisher.publish()`，也不代表 subscriber 收到訊息。不可把舊檔案重新解讀為 schema version 2。

每筆成功的 policy command 以 `command_profile.control_trace_sequence` 對應 tracer sequence；`control_trace_status` 記錄 command 提交時觀察到的狀態：

- `pending`：必要的 `SensorTiming` 尚未齊全，trace 仍在佇列中。
- `publisher_called`：`ControlTracer` 已呼叫 ROS publisher；這不代表 subscriber 已收到訊息。
- `not_submitted`：policy inference 在提交 trace 前失敗；sequence 為 null。

Sidecar 另以 `trace_sequence` 關聯 `control_trace_state` 事件。`pending` 事件表示已提交給 tracer；後續的 `publisher_called` 事件表示呼叫 publisher，並記錄原因（`required_timings_available` 或 `deadline_expired`）。Trace 可先提交並持續 pending；command row 與狀態事件以 sequence 關聯。

Reset row 包含 `discarded_pending_count` 與 `discarded_pending_trace_sequences`。數量大於零時，`discard_reason` 為 `reset_cleared_pending`。這些欄位只記錄既有 reset 行為，不會 flush pending trace、改變 sequence 編號或改動 runtime reset／fresh-sample barrier。Profiler 關閉時維持既有 publish、timeout 與 reset 行為，也不會寫 sidecar。
