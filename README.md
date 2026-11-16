# qwen25-omni-alert

使用 Qwen2.5-Omni-3B 读取水库监控视频（画面 + 音轨），发现有人下水游泳时，由模型通过函数调用触发告警。

## 运行

```bash
source .venv/bin/activate
```

### 水库监控告警（omni_alert.py）

视频理解 + 函数调用告警闭环：

```bash
python omni_alert.py                        # 默认分析 assets/ 下全部 mp4
python omni_alert.py assets/水库监控2.mp4    # 指定视频
```

输出：每个视频打印检测结论；有人下水时进入告警段，模型调用 `send_alert`
并回读告警编号，结尾汇总各视频状态与告警次数。

### 视频理解 + 语音生成（omni_draw.py）

读取 `assets/draw.mp4`（画面）+ `assets/draw_audio.wav`（音轨），模型边看边听，
生成文本回复与对应语音：

```bash
python omni_draw.py
```

输出：终端打印回复文本，并在当前目录生成 `output.wav`（24kHz 单声道语音）。

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

- 模型：`/data/models/Qwen2.5-Omni-3B`（bf16，16GB 显存可完整放下）
- 依赖：uv 管理，全部固化在 `pyproject.toml` / `uv.lock`，`uv sync` 即可复现：
  - torch/torchvision/torchcodec pin 在 PyTorch 官方 cu128 源（`[[tool.uv.index]]`
    `pytorch-cu128`）：本机驱动 570.x 最高支持 CUDA 12.8，而 torch 2.12+ 只发布
    cu13 wheel，故 torch 2.11.0+cu128 / torchvision 0.26.0+cu128 / torchcodec
    0.11.1+cu128（torchcodec 0.11 ↔ torch 2.11 兼容矩阵）；
  - `qwen-omni-utils`、`librosa`、`av`、`audioread` 已入 lockfile，无需手动补装。
- 实测要点（详见 `../qwen25-omni-demo/CLAUDE.md`）：
  - **torchcodec 需要垫片**：cu128 wheel 链接了 NPP 12.8.3+ 的新符号（NV12
    ColorTwist 变体），公开渠道的 NPP（pip 最高 12.4 / 系统 CUDA toolkit）都没有，
    import 即报 undefined symbol。这些符号仅 GPU 解码路径使用，`omni_compat.py`
    用 dlopen(RTLD_LAZY) 预加载绕过（CPU 解码不受影响），两个脚本 import
    qwen_omni_utils 前都会调用它；
  - torchvision ≥0.29 移除了 `io.read_video`，qwen_omni_utils 读视频需要 torchcodec；
  - `librosa 1.x` 移除了 audioread 后端，视频音轨先用 ffmpeg 预提取 16kHz 单声道 wav；
  - transformers 5.22+ 中视觉输入要加 `thinker_` 前缀路由，否则会传给 talker 报错；
  - `<tool_call>`/裸 JSON 是该模型两种工具调用方言，解析器都收。

## 文件

- `omni_alert.py` — 用例主脚本（视频输入 + 函数调用告警）
- `omni_draw.py` — Omni 视频+音频理解与语音输出用例（画视频，输出 output.wav）
- `omni_compat.py` — torchcodec LAZY 预加载垫片（见上文实测要点）
- `main.py` — 纯文本版函数调用闭环（Qwen2.5-3B-Instruct，参考实现）
- `demo.py` — Omni 视频理解的原始 demo（音频输出轨）
