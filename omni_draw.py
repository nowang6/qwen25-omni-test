from pathlib import Path

import soundfile as sf

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

model = Qwen2_5OmniForConditionalGeneration.from_pretrained("/data/models/Qwen2.5-Omni-3B", torch_dtype="auto", device_map="auto")


from qwen_omni_utils import process_mm_info

processor = Qwen2_5OmniProcessor.from_pretrained("/data/models/Qwen2.5-Omni-3B")

# av 无法直接流式读取远程视频 URL，先下载到本地（assets/draw.mp4）
VIDEO_PATH = str(Path(__file__).parent / "assets" / "draw.mp4")
# librosa 1.x 移除了 audioread 后端，qwen_omni_utils 的 use_audio_in_video 路径不可用，
# 改用 ffmpeg 预提取音轨（16kHz 单声道），作为独立 audio 输入
AUDIO_PATH = str(Path(__file__).parent / "assets" / "draw_audio.wav")

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            # 4K 长视频显存/上下文都吃不消：fps 2.0→1.0（21 帧），
            # 并在下方限制每帧 max_pixels，把视觉 token 总量压到 ~1.6 万
            {"type": "video", "video": VIDEO_PATH, "fps": 1.0},
            {"type": "audio", "audio": AUDIO_PATH},
        ],
    },
]

# set use audio in video
# 音轨已作为独立 audio 输入，不再从视频里二次提取
USE_AUDIO_IN_VIDEO = False

# Preparation for inference
# 原 max_pixels=12.8M 等于单帧 4K 全保真（~1 万 patch/帧），21 帧就是 40 多万 token，
# 16GB 显存 + 32k 上下文都放不下；压到 589824 px ≈ 752 patch/帧，总量 ~1.6 万 token
processor.image_processor.max_pixels = 589_824
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)
# transformers 5.22 的 bug：视觉输入会落入 shared_kwargs 被原样传给内部 talker.generate，
# 触发 "model_kwargs are not used by the model: ['pixel_values_videos']"。
# 加 thinker_ 前缀可只把视觉输入路由给 thinker。
inputs = {
    ("thinker_" + k if k in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw") else k): v
    for k, v in inputs.items()
}
input_len = inputs["input_ids"].shape[1]

# Inference: Generation of the output text and audio
# 注意（transformers 5.x）：
# - 音频输出必须显式传 generation_mode="audio"，返回 (text_ids, waveform) 元组
# - text_ids 含 prompt，要 [:, input_len:] 再解码，否则会打出 token ID 列表
# - talker 默认 top_k=40/top_p=0.8 会让 codec eos 数学上不可达（见 runtime/engine.py），这里放开采样
text_ids, audio = model.generate(
    **inputs,
    speaker="Chelsie",
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
    generation_mode="audio",
    thinker_max_new_tokens=512,
    talker_max_new_tokens=1000,
    talker_top_k=0,
    talker_top_p=1.0,
    talker_temperature=1.0,
)

text = processor.batch_decode(text_ids[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("text: ", text)
sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
print("saved output.wav")

