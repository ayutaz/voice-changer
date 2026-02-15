import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from data.ModelSlot import RVCModelSlot, loadSlotInfo
from voice_changer.RVC.RVCr2 import RVCr2
from voice_changer.VoiceChangerV2 import VoiceChangerV2
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


def _build_params(model_dir: str, sample_mode: str) -> VoiceChangerParams:
    return VoiceChangerParams(
        model_dir=model_dir,
        content_vec_500="pretrain/checkpoint_best_legacy_500.pt",
        content_vec_500_onnx="pretrain/content_vec_500.onnx",
        content_vec_500_onnx_on=True,
        hubert_base="pretrain/hubert_base.pt",
        hubert_base_jp="pretrain/rinna_hubert_base_jp.pt",
        hubert_soft="pretrain/hubert/hubert-soft-0d54a1f4.pt",
        nsf_hifigan="pretrain/nsf_hifigan/model",
        sample_mode=sample_mode,
        crepe_onnx_full="pretrain/crepe_onnx_full.onnx",
        crepe_onnx_tiny="pretrain/crepe_onnx_tiny.onnx",
        rmvpe="pretrain/rmvpe.pt",
        rmvpe_onnx="pretrain/rmvpe.onnx",
        whisper_tiny="pretrain/whisper_tiny.pt",
    )


def _to_int16_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)
    else:
        audio = audio.astype(np.int16)
    return audio


def run_offline_conversion(
    input_wav: Path,
    output_wav: Path,
    model_dir: str,
    slot_index: int,
    chunk_size: int,
    sample_mode: str,
) -> None:
    params = _build_params(model_dir=model_dir, sample_mode=sample_mode)
    slot = loadSlotInfo(model_dir, slot_index)
    if not isinstance(slot, RVCModelSlot):
        raise RuntimeError(f"slot {slot_index} is not RVC")
    slot.slotIndex = slot_index

    model = RVCr2(params, slot)
    vc = VoiceChangerV2(params)
    vc.setModel(model)

    audio, sr = sf.read(str(input_wav), always_2d=False)
    audio_i16 = _to_int16_mono(audio)

    vc.update_settings("inputSampleRate", int(sr))
    vc.update_settings("outputSampleRate", int(sr))
    vc.update_settings("f0Detector", "rmvpe_onnx")
    vc.update_settings("gpu", 0)

    outputs: list[np.ndarray] = []
    for start in range(0, len(audio_i16), chunk_size):
        chunk = audio_i16[start : start + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        converted, _perf = vc.on_request(chunk)
        outputs.append(converted)

    out = np.concatenate(outputs).astype(np.int16)
    out = out[: len(audio_i16)]
    sf.write(str(output_wav), out, int(sr), subtype="PCM_16")


def _parse_args():
    parser = argparse.ArgumentParser(description="Offline file-based VC test using RVC pipeline.")
    parser.add_argument("--input-wav", type=str, required=True)
    parser.add_argument("--output-wav", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default="model_dir_uv_testofficial")
    parser.add_argument("--slot-index", type=int, default=4, help="Use ONNX sample slot by default")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--sample-mode", type=str, default="testOfficial")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_offline_conversion(
        input_wav=Path(args.input_wav),
        output_wav=Path(args.output_wav),
        model_dir=args.model_dir,
        slot_index=args.slot_index,
        chunk_size=args.chunk_size,
        sample_mode=args.sample_mode,
    )
    print(f"[offline_vc_test] done: {args.output_wav}")


if __name__ == "__main__":
    main()
