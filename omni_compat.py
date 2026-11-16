"""torchcodec 在本机的兼容垫片（必须在 import qwen_omni_utils / torchcodec 之前调用）。

问题：torchcodec cu128 wheel（libtorchcodec_coreN.so）链接了 NPP 12.8.3+ 才有的
新符号（nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx 等 NV12 ColorTwist 变体，见
readelf --dyn-syms），而公开渠道能装到的 NPP 均无此符号：
  - PyPI / pypi.nvidia.com 的 nvidia-npp-cu12 最高 12.4.1.87；
  - 本机 /usr/local/cuda-12.{1,2,8} toolkit 的 NPP 也没有。
这些符号只被 GPU 解码路径调用；CPU 解码（qwen_omni_utils 的路径）完全不用。

原理（glibc）：dlopen(RTLD_LAZY) 允许推迟函数符号重定位，NEEDED 依赖库立即加载；
同一库第二次 dlopen 时（torchcodec 内部 torch.ops.load_library 用 RTLD_NOW）
glibc 直接复用已加载的 handle，不再重新解析符号。于是先按系统 FFmpeg 版本挑一个
libtorchcodec_coreN.so 用 LAZY 预加载，torchcodec 即可正常 import。

用法：
    import omni_compat; omni_compat.patch_torchcodec()
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

_RTLazy = 0x1  # Linux RTLD_LAZY（ctypes 未暴露该常量）


def patch_torchcodec() -> None:
    """让 torchcodec 可 import。环境已可用时是 no-op（不受未来修复影响）。"""
    try:
        import torchcodec  # noqa: F401
        return
    except Exception:
        pass

    import torch  # 先加载 libtorch.so（core 库 NEEDED 它，且需 RTLD_GLOBAL）
    torchcodec_dir = Path(torch.__file__).parent.parent / "torchcodec"
    for ffmpeg_ver in range(4, 9):  # FFmpeg 4~8，依赖齐全的那套才 LAZY 加载得动
        so = torchcodec_dir / f"libtorchcodec_core{ffmpeg_ver}.so"
        if not so.exists():
            continue
        try:
            ctypes.CDLL(str(so), mode=_RTLazy | os.RTLD_GLOBAL)
            return
        except OSError:
            continue
    raise RuntimeError(
        "torchcodec 兼容垫片失败：没有可预加载的 libtorchcodec_coreN.so"
        "（请检查 torchcodec 是否随 venv 正常安装）"
    )
