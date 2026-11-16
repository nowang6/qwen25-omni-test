"""水库监控告警用例：Qwen2.5-Omni-3B 读取监控视频，发现有人游泳则函数调用告警。

两段式流程（实测比单段 prompt 驱动更稳，见下）：
  1. 检测段：omni thinker 读取监控视频（画面 + 音轨），用自然语言描述
     「是否有人下水游泳」。描述式回答对 3B 模型最可靠。
  2. 告警段：程序判断描述为阳性时，发起一次纯文本对话，让模型按
     <tool_call> 格式调用 send_alert —— 函数调用由模型真正发出，程序解析后
     执行，并把工具结果回传模型做确认（与 main.py 的工具闭环同构）。

为什么不单段搞定：实测 3B omni 模型要么看完视频口头报告「有人游泳」却忘了
输出 tool_call，要么在 few-shot 示例下照搬最后一个「画面正常」示例——
单段同时承担「看懂视频」和「按格式调工具」两件事时格式遵从很不稳定，
拆成两段后各自都可靠。

环境注意（详见 ../qwen25-omni-demo 的实测结论）：
  - transformers 5.22 中视觉输入会落入 shared_kwargs 被传给内部 talker.generate 报错，
    加 thinker_ 前缀只路由给 thinker（本脚本只用文本轨，保留前缀以便切换到音频轨）；
  - librosa 1.x 移除了 audioread 后端，use_audio_in_video 不可用，
    先用 ffmpeg 预提取 16kHz 单声道音轨作为独立 audio 输入；
  - torchvision 0.29 移除了 io.read_video，qwen_omni_utils 需要 torchcodec 读视频；
  - 4K/高帧率视频会爆显存与上下文，限制 fps 与每帧 max_pixels。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

MODEL_PATH = "/data/models/Qwen2.5-Omni-3B"
DEFAULT_VIDEO_DIR = Path(__file__).parent / "asserts"
MAX_STEPS = 3

#: 视觉输入的 processor 输出键：transformers 5.22 里它们落入 shared_kwargs 会被
#: 原样传给内部 talker.generate（"model_kwargs are not used"），加 thinker_ 前缀规避
_VISUAL_KEYS = ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw")

# ---------------------------------------------------------------- 系统提示

ANALYSIS_SYSTEM_PROMPT = """你是水库安防监控 AI。仔细观察监控画面（含环境音），用一两句话回答：
水库水域中是否有人游泳或戏水？如有，说明位置和人数；如没有，说明画面正常。
只输出描述，不要输出 JSON、工具调用或其他格式。"""

ALERT_SYSTEM_PROMPT = """你是水库监控告警程序。可用工具：send_alert(location: str, detail: str) —— 有人下水游泳时发送告警。

值班员从监控画面确认：{summary}

请调用 send_alert 发送告警：在回复开头输出工具调用 JSON（可以包在 <tool_call></tool_call> 中，也可以只输出 JSON 本体），收到工具返回结果后，向值班员报告告警编号。

调用格式：
<tool_call>{{"name": "send_alert", "arguments": {{"location": "<泳者在画面中的位置>", "detail": "<人数与动作>"}}}}</tool_call>"""

ALERT_COUNT = 0


def send_alert(location: str = "", detail: str = "") -> dict:
    """告警工具：发送告警通知。"""
    global ALERT_COUNT
    ALERT_COUNT += 1
    alert_id = f"ALT-{ALERT_COUNT:04d}"
    print(f"\n🚨  告警工具被调用！告警编号: {alert_id}")
    if location:
        print(f"   位置: {location}")
    if detail:
        print(f"   详情: {detail}")
    print()
    return {"status": "success", "alert_id": alert_id, "location": location, "detail": detail}


TOOL_FUNCTIONS = {"send_alert": send_alert}

# ---------------------------------------------------------------- 工具调用解析


def _balanced_json_objects(text: str) -> list[str]:
    """用花括号配平扫描提取 JSON 对象文本（嵌套 arguments 也能完整取出）。"""
    objects: list[str] = []
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                objects.append(text[start : i + 1])
    return objects


def extract_tool_call(text: str) -> dict | None:
    """从模型输出中解析工具调用 JSON。

    实测 Qwen2.5-Omni-3B 偏好直接输出裸 JSON（不包 <tool_call> 特殊 token），
    这里两种方言都收；JSON 按花括号配平提取，兼容嵌套的 arguments。
    """
    candidates: list[str] = []
    # 1) <tool_call>...</tool_call>（与 main.py 同构；解码保留 special tokens 时命中）
    candidates += re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
    # 2) ```json 代码块
    candidates += re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    # 3) 全文配平扫描（裸 JSON 方言 + 兜底）
    candidates += _balanced_json_objects(text)

    for raw in candidates:
        if '"name"' not in raw:
            continue
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict) and obj.get("name") in TOOL_FUNCTIONS:
            args = obj.get("arguments", {})
            return {"name": obj["name"], "arguments": args if isinstance(args, dict) else {}}
    return None


# ---------------------------------------------------------------- 音频预处理


def extract_audio(video_path: Path) -> str | None:
    """ffmpeg 预提取 16kHz 单声道音轨（音频编码器原生采样率，见 ../qwen25-omni-demo）。

    返回临时 wav 路径；视频无音轨时返回 None。
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    if "audio" not in probe.stdout:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video_path),
         "-vn", "-ac", "1", "-ar", "16000", tmp.name],
        check=True,
    )
    return tmp.name


# ---------------------------------------------------------------- 推理封装


def encode_inputs(processor, conversation: list[dict], model) -> dict:
    """消息列表 → 模型输入（视觉键加 thinker_ 前缀，见模块注释）。"""
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    return {
        ("thinker_" + k if k in _VISUAL_KEYS else k): v for k, v in inputs.items()
    }


def generate_text(model, processor, inputs: dict, max_new_tokens: int = 256) -> str:
    """文本轨生成（generation_mode="text" 返回裸 token 张量，含 prompt 需裁剪）。"""
    input_len = inputs["input_ids"].shape[1]
    output = model.generate(
        **inputs,
        generation_mode="text",
        thinker_max_new_tokens=max_new_tokens,
    )
    return processor.batch_decode(
        output[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def _looks_like_swimming_detected(text: str) -> bool:
    """检测段的描述是否判定为「有人下水」。程序依据它决定是否进入告警段。"""
    positive = any(w in text for w in ("有人", "一人", "两名", "三人", "多名")) and any(
        w in text for w in ("游泳", "戏水", "下水", "水域中")
    )
    negative = any(w in text for w in ("无人", "没有", "未发现", "正常"))
    return positive and not negative


def monitor_video(model, processor, video_path: Path) -> dict:
    """两段式处理一段监控视频：检测段（看视频）→ 告警段（模型函数调用闭环）。"""
    print("=" * 78)
    print(f"📹 监控视频: {video_path.name}")
    print("=" * 78)

    audio_path = extract_audio(video_path)
    user_content = [{"type": "video", "video": str(video_path), "fps": 2.0}]
    if audio_path:
        user_content.append({"type": "audio", "audio": audio_path})
    user_content.append(
        {"type": "text", "text": "请分析这段水库监控录像，是否有人下水游泳或戏水？"}
    )

    # ---------------- 检测段：描述式回答，对 3B 模型最可靠
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": ANALYSIS_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
    inputs = encode_inputs(processor, conversation, model)
    description = generate_text(model, processor, inputs)
    print("--- 检测结论 ---")
    print(description)

    if not _looks_like_swimming_detected(description):
        return {"video": video_path.name, "alerted": False, "alerts": [],
                "final_reply": description.strip()}

    # ---------------- 告警段：阳性判定后，由模型发出函数调用（与 main.py 闭环同构）
    print("\n（判定有人下水，进入告警段）")
    conversation = [
        {"role": "system", "content": [
            {"type": "text", "text": ALERT_SYSTEM_PROMPT.format(summary=description.strip())}
        ]},
        {"role": "user", "content": [
            {"type": "text", "text": "请根据以上确认信息立即调用 send_alert 发送告警。"}
        ]},
    ]

    alerts: list[dict] = []
    final_reply = ""
    for step in range(MAX_STEPS):
        inputs = encode_inputs(processor, conversation, model)
        response = generate_text(model, processor, inputs)
        print(f"--- 模型输出 (告警段 step {step + 1}) ---")
        print(response)
        conversation.append({"role": "assistant", "content": response})

        tool_call = extract_tool_call(response)
        if tool_call:
            name, args = tool_call["name"], tool_call["arguments"]
            result = TOOL_FUNCTIONS[name](**args)
            alerts.append(result)
            conversation.append({
                "role": "tool",
                "name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })
            continue
        if alerts:
            final_reply = response.strip()  # 工具结果之后的最终确认
            break
        # 模型还没出工具调用：按格式再提醒一次
        conversation.append({"role": "user", "content": [
            {"type": "text",
             "text": "还没有看到 send_alert 工具调用。请在回复开头输出"
                     " {\"name\": \"send_alert\", \"arguments\": {...}} 格式的工具调用"
                     "（可包在 <tool_call></tool_call> 中），收到工具返回后再确认。"}
        ]})
    else:
        print(f"（达到最大迭代次数 {MAX_STEPS}，停止）")

    return {"video": video_path.name, "alerted": bool(alerts), "alerts": alerts,
            "final_reply": final_reply}


def main() -> None:
    parser = argparse.ArgumentParser(description="水库游泳监测：Omni 视频理解 + 函数调用告警")
    parser.add_argument(
        "videos", nargs="*", type=Path,
        help="监控视频路径，默认分析 asserts/ 下全部 mp4",
    )
    args = parser.parse_args()
    videos = args.videos or sorted(DEFAULT_VIDEO_DIR.glob("*.mp4"))
    if not videos:
        sys.exit(f"未找到监控视频：{DEFAULT_VIDEO_DIR}")

    print(f"加载模型: {MODEL_PATH}")
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="cuda",
        # 注意不能关 talker：transformers 5.23 的 generate() 构造 talker_kwargs 时
        # 无条件访问 self.talker.codec_pad_token，关了会在文本轨直接 AttributeError
    )
    processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_PATH)
    # 视觉 token 总量控制：589824 px ≈ 752 patch/帧，10 帧（fps 2×5s）约 8k token
    processor.image_processor.max_pixels = 589_824

    results = [monitor_video(model, processor, v) for v in videos]

    print("=" * 78)
    print("📋 汇总")
    print("=" * 78)
    for r in results:
        status = "🚨 已告警" if r["alerted"] else "✅ 正常"
        ids = ", ".join(a["alert_id"] for a in r["alerts"]) or "-"
        print(f"  {r['video']:<24s} {status}  告警: {ids}")
    print(f"\n共发送告警 {ALERT_COUNT} 次")


if __name__ == "__main__":
    main()
