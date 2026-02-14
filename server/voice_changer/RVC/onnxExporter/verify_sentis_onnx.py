import argparse
import json
import os
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

RISKY_OP_HINTS = {
    "NonZero": "CPU-only in Sentis.",
    "TopK": "Not supported on GPUPixel in Sentis.",
    "Shape": "Handled as a CPU tensor in Sentis and can increase transfer overhead.",
}

ORT_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int8)": np.int8,
    "tensor(int16)": np.int16,
    "tensor(int32)": np.int32,
    "tensor(int64)": np.int64,
    "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16,
    "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64,
    "tensor(bool)": np.bool_,
}


def _load_metadata(model: onnx.ModelProto) -> dict[str, Any]:
    for prop in model.metadata_props:
        if prop.key != "metadata":
            continue
        try:
            value = json.loads(prop.value)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            return {}
    return {}


def _collect_int64_usages(model: onnx.ModelProto) -> list[str]:
    issues: list[str] = []

    for value in model.graph.input:
        if value.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            issues.append(f"graph.input:{value.name}")
    for value in model.graph.output:
        if value.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            issues.append(f"graph.output:{value.name}")
    for value in model.graph.value_info:
        if value.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            issues.append(f"graph.value_info:{value.name}")
    for initializer in model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.INT64:
            issues.append(f"graph.initializer:{initializer.name}")
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == onnx.TensorProto.INT64:
                    issues.append(f"node.Constant:{node.name or '<unnamed>'}")
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == onnx.TensorProto.INT64:
                    issues.append(f"node.Cast:{node.name or '<unnamed>'}")

    return issues


def _resolve_dim(
    dim: Any,
    axis: int,
    rank: int,
    input_name: str,
    seq_len: int,
    batch_size: int,
    emb_channels: int,
) -> int:
    if isinstance(dim, int) and dim > 0:
        return dim

    if input_name in ("p_len", "sid"):
        return batch_size

    if rank == 1:
        return batch_size
    if rank == 2:
        return batch_size if axis == 0 else seq_len
    if rank == 3:
        if axis == 0:
            return batch_size
        if axis == 1:
            return seq_len
        return emb_channels

    return 1


def _build_dummy_input(
    node_arg: ort.NodeArg,
    rng: np.random.Generator,
    seq_len: int,
    batch_size: int,
    emb_channels: int,
) -> np.ndarray:
    dtype = ORT_TYPE_TO_NUMPY.get(node_arg.type)
    if dtype is None:
        raise RuntimeError(f"Unsupported ORT input type: {node_arg.name} ({node_arg.type})")

    raw_shape = node_arg.shape
    shape: list[int] = []
    rank = len(raw_shape)
    for axis, dim in enumerate(raw_shape):
        shape.append(_resolve_dim(dim, axis, rank, node_arg.name, seq_len, batch_size, emb_channels))

    if node_arg.name == "p_len":
        return np.full((batch_size,), seq_len, dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        return np.zeros(shape, dtype=dtype)
    if np.issubdtype(dtype, np.bool_):
        return np.zeros(shape, dtype=dtype)
    return rng.standard_normal(shape).astype(dtype)


def _collect_risky_ops(model: onnx.ModelProto) -> list[str]:
    ops = sorted({node.op_type for node in model.graph.node})
    warnings: list[str] = []
    for op in ops:
        hint = RISKY_OP_HINTS.get(op)
        if hint is not None:
            warnings.append(f"{op}: {hint}")
    return warnings


def _detect_main_opset(model: onnx.ModelProto) -> int | None:
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return int(opset.version)
    return None


def verify_sentis_onnx(
    onnx_path: str,
    max_opset: int = 15,
    run_ort: bool = True,
    seq_len: int = 64,
    batch_size: int = 1,
    seed: int = 0,
) -> bool:
    if not os.path.isfile(onnx_path):
        print(f"[Voice Changer] verify failed: file not found: {onnx_path}")
        return False

    model = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(model)
    except Exception as e:  # noqa: BLE001
        print(f"[Voice Changer] verify failed: onnx.checker.check_model error: {e}")
        return False

    opset = _detect_main_opset(model)
    if opset is None:
        print("[Voice Changer] verify failed: unable to detect main ONNX opset")
        return False
    if opset > max_opset:
        print(
            "[Voice Changer] verify failed:",
            f"opset={opset} exceeds allowed max_opset={max_opset}",
        )
        return False

    int64_issues = _collect_int64_usages(model)
    if int64_issues:
        print("[Voice Changer] verify failed: INT64 usages remain in graph:")
        for item in int64_issues[:20]:
            print(f"  - {item}")
        if len(int64_issues) > 20:
            print(f"  - ... and {len(int64_issues) - 20} more")
        return False

    warnings = _collect_risky_ops(model)
    for warning in warnings:
        print(f"[Voice Changer] verify warning: {warning}")

    print(
        "[Voice Changer] static verification passed:",
        f"opset={opset}",
        f"inputs={len(model.graph.input)}",
        f"outputs={len(model.graph.output)}",
        f"nodes={len(model.graph.node)}",
    )

    if not run_ort:
        return True

    metadata = _load_metadata(model)
    emb_channels = int(metadata.get("embChannels", 256))

    rng = np.random.default_rng(seed)
    try:
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        feeds: dict[str, np.ndarray] = {}
        for node_arg in session.get_inputs():
            feeds[node_arg.name] = _build_dummy_input(
                node_arg=node_arg,
                rng=rng,
                seq_len=seq_len,
                batch_size=batch_size,
                emb_channels=emb_channels,
            )
        outputs = session.run(None, feeds)
    except Exception as e:  # noqa: BLE001
        print(f"[Voice Changer] verify failed: ORT dry-run error: {e}")
        return False

    print("[Voice Changer] ORT dry-run passed:")
    for output_info, value in zip(session.get_outputs(), outputs):
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        print(f"  - {output_info.name}: shape={shape}, dtype={dtype}")

    return True


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Verify Unity Sentis-ready ONNX: opset/int64 checks and optional ORT dry-run."
    )
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX file")
    parser.add_argument("--max-opset", type=int, default=15, help="Allowed max ONNX opset version")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length for ORT dry-run dummy input")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for ORT dry-run dummy input")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for dummy input generation")
    parser.add_argument("--skip-ort", action="store_true", help="Run static checks only")
    return parser.parse_args()


def main():
    args = _parse_args()
    ok = verify_sentis_onnx(
        onnx_path=args.onnx,
        max_opset=args.max_opset,
        run_ort=not args.skip_ort,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    if not ok:
        raise SystemExit(1)
    print("[Voice Changer] verify succeeded.")


if __name__ == "__main__":
    main()
