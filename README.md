# qwen25-omni-alert

使用 Qwen2.5-Omni-3B 读取水库监控视频（画面 + 音轨），发现有人下水游泳时，由模型通过函数调用触发告警。

## 运行

```bash
source .venv/bin/activate
python omni_alert.py                        # 默认分析 asserts/ 下全部 mp4
python omni_alert.py asserts/水库监控2.mp4   # 指定视频
```

## 工作流程

两段式（对 3B 小模型比单段 prompt 驱动更稳）：

1. **检测段**：Omni thinker 看视频/听声音，用自然语言回答「是否有人下水游泳」；
2. **告警段**：程序判定为阳性后发起纯文本对话，模型输出 `send_alert` 工具调用
   （`<tool_call>{...}</tool_call>` 或裸 JSON 方言都支持），程序解析并真正执行
   告警函数，再把工具结果回传模型确认告警编号。

预期结果（与抽帧真值一致）：

| 视频 | 画面内容 | 结果 |
|---|---|---|
| 水库监控1.mp4 | 空库岸，无人 | ✅ 正常，不告警 |
| 水库监控2.mp4 | 一人游泳 | 🚨 已告警 ALT-xxxx |

## 环境

- 模型：`/data/models/Qwen2.5-Omni-3B`（bf16，RTX 5060 Ti 16GB 可完整放下）
- 依赖：uv 管理，见 `pyproject.toml` / `uv.lock`；另手动 `uv pip install` 了
  `qwen-omni-utils`、`librosa`、`av`、`torchcodec`、`audioread`（不在 lockfile 里，
  `uv sync` 后需补装）
- 实测要点（详见 `../qwen25-omni-demo/CLAUDE.md`）：
  - torchvision ≥0.29 移除了 `io.read_video`，qwen_omni_utils 读视频需要 torchcodec；
  - `librosa 1.x` 移除了 audioread 后端，视频音轨先用 ffmpeg 预提取 16kHz 单声道 wav；
  - transformers 5.22+ 中视觉输入要加 `thinker_` 前缀路由，否则会传给 talker 报错；
  - `<tool_call>`/裸 JSON 是该模型两种工具调用方言，解析器都收。

## 文件

- `omni_alert.py` — 用例主脚本（视频输入 + 函数调用告警）
- `main.py` — 纯文本版函数调用闭环（Qwen2.5-3B-Instruct，参考实现）
- `demo.py` — Omni 视频理解的原始 demo（音频输出轨）
