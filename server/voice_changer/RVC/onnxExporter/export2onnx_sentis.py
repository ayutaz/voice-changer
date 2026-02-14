import argparse
import json
import os
from typing import Any

import numpy as np
import onnx
import torch
from onnxsim import simplify

from const import TMP_DIR, EnumInferenceTypes
from data.ModelSlot import RVCModelSlot, loadSlotInfo
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.onnxExporter.SynthesizerTrnMs256NSFsid_ONNX import (
    SynthesizerTrnMs256NSFsid_ONNX,
)
from voice_changer.RVC.onnxExporter.SynthesizerTrnMs256NSFsid_nono_ONNX import (
    SynthesizerTrnMs256NSFsid_nono_ONNX,
)
from voice_changer.RVC.onnxExporter.SynthesizerTrnMs768NSFsid_ONNX import (
    SynthesizerTrnMs768NSFsid_ONNX,
)
from voice_changer.RVC.onnxExporter.SynthesizerTrnMs768NSFsid_nono_ONNX import (
    SynthesizerTrnMs768NSFsid_nono_ONNX,
)
from voice_changer.RVC.onnxExporter.SynthesizerTrnMsNSFsidNono_webui_ONNX import (
    SynthesizerTrnMsNSFsidNono_webui_ONNX,
)
from voice_changer.RVC.onnxExporter.SynthesizerTrnMsNSFsid_webui_ONNX import (
    SynthesizerTrnMsNSFsid_webui_ONNX,
)
from voice_changer.RVC.onnxExporter.verify_sentis_onnx import verify_sentis_onnx
from voice_changer.VoiceChangerParamsManager import VoiceChangerParamsManager


def _create_model_for_export(cpt: dict[str, Any], model_type: str, is_half: bool):
    if model_type == EnumInferenceTypes.pyTorchRVC.value:
        return SynthesizerTrnMs256NSFsid_ONNX(*cpt["config"], is_half=is_half)
    if model_type == EnumInferenceTypes.pyTorchWebUI.value:
        return SynthesizerTrnMsNSFsid_webui_ONNX(**cpt["params"], is_half=is_half)
    if model_type == EnumInferenceTypes.pyTorchRVCNono.value:
        return SynthesizerTrnMs256NSFsid_nono_ONNX(*cpt["config"])
    if model_type == EnumInferenceTypes.pyTorchWebUINono.value:
        return SynthesizerTrnMsNSFsidNono_webui_ONNX(**cpt["params"])
    if model_type == EnumInferenceTypes.pyTorchRVCv2.value:
        return SynthesizerTrnMs768NSFsid_ONNX(*cpt["config"], is_half=is_half)
    if model_type == EnumInferenceTypes.pyTorchRVCv2Nono.value:
        return SynthesizerTrnMs768NSFsid_nono_ONNX(*cpt["config"])

    raise ValueError(f"Unsupported RVC modelType for export: {model_type}")


def _convert_tensor_int64_to_int32(tensor: onnx.TensorProto) -> bool:
    if tensor.data_type != onnx.TensorProto.INT64:
        return False

    converted = False
    if tensor.raw_data:
        arr = np.frombuffer(tensor.raw_data, dtype=np.int64).astype(np.int32)
        tensor.raw_data = arr.tobytes()
        converted = True
    elif tensor.int64_data:
        arr = np.asarray(tensor.int64_data, dtype=np.int64).astype(np.int32)
        del tensor.int64_data[:]
        tensor.int32_data.extend(arr.tolist())
        converted = True

    tensor.data_type = onnx.TensorProto.INT32
    return converted


def convert_model_int64_to_int32(model: onnx.ModelProto) -> onnx.ModelProto:
    converted = 0

    for t in model.graph.input:
        if t.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            t.type.tensor_type.elem_type = onnx.TensorProto.INT32
            converted += 1

    for t in model.graph.output:
        if t.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            t.type.tensor_type.elem_type = onnx.TensorProto.INT32
            converted += 1

    for t in model.graph.value_info:
        if t.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            t.type.tensor_type.elem_type = onnx.TensorProto.INT32
            converted += 1

    for initializer in model.graph.initializer:
        if _convert_tensor_int64_to_int32(initializer):
            converted += 1

    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and _convert_tensor_int64_to_int32(attr.t):
                    converted += 1
        elif node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == onnx.TensorProto.INT64:
                    attr.i = onnx.TensorProto.INT32
                    converted += 1

    print(f"[Voice Changer] int64->int32 converted nodes/tensors: {converted}")
    return model


def _generate_output_names(model_file: str, output_dir: str, use_fp16: bool) -> tuple[str, str, str]:
    stem = os.path.splitext(os.path.basename(model_file))[0]
    suffix = "_sentis_op15_fp16" if use_fp16 else "_sentis_op15_fp32"
    output_file = f"{stem}{suffix}.onnx"
    output_file_simple = f"{stem}{suffix}_simple.onnx"
    output_path = os.path.join(output_dir, output_file)
    output_path_simple = os.path.join(output_dir, output_file_simple)
    return output_file, output_path, output_path_simple


def _export_checkpoint(
    input_model: str,
    output_model: str,
    output_model_simple: str,
    is_half: bool,
    metadata: dict[str, Any],
    opset_version: int = 15,
):
    cpt = torch.load(input_model, map_location="cpu")
    dev = torch.device("cuda", index=0) if is_half else torch.device("cpu")

    net_g_onnx = _create_model_for_export(cpt, metadata["modelType"], is_half)
    net_g_onnx.eval().to(dev)
    net_g_onnx.load_state_dict(cpt["weight"], strict=False)
    if is_half:
        net_g_onnx = net_g_onnx.half()

    feats_length = 64
    if is_half:
        feats = torch.HalfTensor(1, feats_length, metadata["embChannels"]).to(dev)
    else:
        feats = torch.FloatTensor(1, feats_length, metadata["embChannels"]).to(dev)

    p_len = torch.LongTensor([feats_length]).to(dev)
    sid = torch.LongTensor([0]).to(dev)

    if metadata["f0"] is True:
        pitch = torch.zeros(1, feats_length, dtype=torch.int64).to(dev)
        pitchf = torch.FloatTensor(1, feats_length).to(dev)
        input_names = ["feats", "p_len", "pitch", "pitchf", "sid"]
        inputs = (feats, p_len, pitch, pitchf, sid)
        dynamic_axes = {
            "feats": [1],
            "pitch": [1],
            "pitchf": [1],
        }
    else:
        input_names = ["feats", "p_len", "sid"]
        inputs = (feats, p_len, sid)
        dynamic_axes = {"feats": [1]}

    torch.onnx.export(
        net_g_onnx,
        inputs,
        output_model,
        dynamic_axes=dynamic_axes,
        do_constant_folding=False,
        opset_version=opset_version,
        verbose=False,
        input_names=input_names,
        output_names=["audio"],
    )

    model_onnx = onnx.load(output_model)
    try:
        model_simp, check = simplify(model_onnx)
        if not check:
            print("[Voice Changer] Warning: onnxsim check failed. keeping simplified model.")
    except Exception as e:  # noqa: BLE001
        print(f"[Voice Changer] Warning: onnx simplify failed. fallback to raw model. {e}")
        model_simp = model_onnx

    model_int32 = convert_model_int64_to_int32(model_simp)
    meta = model_int32.metadata_props.add()
    meta.key = "metadata"
    meta.value = json.dumps(metadata)
    onnx.save(model_int32, output_model_simple)


def export_checkpoint_to_sentis_onnx(
    checkpoint_path: str,
    model_slot: RVCModelSlot,
    output_dir: str = TMP_DIR,
    use_fp16: bool = False,
    gpu: int = 0,
    opset_version: int = 15,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    use_fp16_effective = use_fp16
    if use_fp16:
        gpu_memory = DeviceManager.get_instance().getDeviceMemory(gpu)
        if gpu_memory <= 0:
            print("[Voice Changer] Warning: fp16 export requested but CUDA unavailable. fallback to fp32.")
            use_fp16_effective = False

    output_file, output_path, output_path_simple = _generate_output_names(
        checkpoint_path, output_dir, use_fp16_effective
    )
    output_file_simple = os.path.basename(output_path_simple)

    metadata = {
        "application": "VC_CLIENT",
        "version": "2.1",
        "runtime": "sentis",
        "opset": opset_version,
        "modelType": model_slot.modelType,
        "samplingRate": model_slot.samplingRate,
        "f0": model_slot.f0,
        "embChannels": model_slot.embChannels,
        "embedder": model_slot.embedder,
        "embOutputLayer": model_slot.embOutputLayer,
        "useFinalProj": model_slot.useFinalProj,
    }

    print(
        "[Voice Changer] exporting sentis onnx...",
        f"checkpoint={checkpoint_path}",
        f"output={output_file}",
        f"fp16={use_fp16_effective}",
        f"opset={opset_version}",
    )
    _export_checkpoint(
        checkpoint_path,
        output_path,
        output_path_simple,
        use_fp16_effective,
        metadata,
        opset_version=opset_version,
    )

    return output_file_simple


def export2onnx_sentis(gpu: int, model_slot: RVCModelSlot, use_fp16: bool = False) -> str:
    vcparams = VoiceChangerParamsManager.get_instance().params
    model_file = os.path.join(
        vcparams.model_dir,
        str(model_slot.slotIndex),
        os.path.basename(model_slot.modelFile),
    )
    return export_checkpoint_to_sentis_onnx(
        model_file,
        model_slot=model_slot,
        output_dir=TMP_DIR,
        use_fp16=use_fp16,
        gpu=gpu,
        opset_version=15,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Export RVC checkpoint to Unity Sentis-compatible ONNX (opset 15 + int32 conversion)."
    )
    parser.add_argument("--slot-index", type=int, required=True, help="RVC slot index")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="logs",
        help="Model directory that contains slot folders and params.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=TMP_DIR,
        help="Directory to save exported ONNX files",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index for optional fp16 export")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 export when CUDA is available")
    parser.add_argument("--verify", action="store_true", help="Run Sentis checks after export")
    parser.add_argument("--verify-max-opset", type=int, default=15, help="Allowed max opset for verification")
    parser.add_argument(
        "--verify-seq-len",
        type=int,
        default=64,
        help="Sequence length used for ORT dry-run verification",
    )
    parser.add_argument(
        "--verify-batch-size",
        type=int,
        default=1,
        help="Batch size used for ORT dry-run verification",
    )
    parser.add_argument(
        "--verify-skip-ort",
        action="store_true",
        help="Skip ORT dry-run and run static checks only",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    slot = loadSlotInfo(args.model_dir, args.slot_index)
    if not isinstance(slot, RVCModelSlot):
        raise RuntimeError(f"slot {args.slot_index} is not RVC")
    if slot.isONNX:
        raise RuntimeError("slot is already ONNX. pytorch checkpoint is required")

    slot.slotIndex = args.slot_index
    checkpoint_path = os.path.join(
        args.model_dir,
        str(args.slot_index),
        os.path.basename(slot.modelFile),
    )
    output_file = export_checkpoint_to_sentis_onnx(
        checkpoint_path=checkpoint_path,
        model_slot=slot,
        output_dir=args.output_dir,
        use_fp16=args.fp16,
        gpu=args.gpu,
        opset_version=15,
    )
    output_path = os.path.join(args.output_dir, output_file)
    print(f"[Voice Changer] Sentis ONNX exported: {output_path}")

    if args.verify:
        ok = verify_sentis_onnx(
            onnx_path=output_path,
            max_opset=args.verify_max_opset,
            run_ort=not args.verify_skip_ort,
            seq_len=args.verify_seq_len,
            batch_size=args.verify_batch_size,
        )
        if not ok:
            raise RuntimeError("Sentis ONNX verification failed.")
        print("[Voice Changer] Sentis ONNX verification passed.")


if __name__ == "__main__":
    main()
