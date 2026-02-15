import numpy as np
import onnxruntime as ort
import torch

from voice_changer.RVC.embedder.Embedder import Embedder


class OnnxContentvec(Embedder):
    def loadModel(
        self,
        file: str,
        dev: torch.device,
        isHalf: bool = True,
        embedderType: str = "hubert_base",
    ) -> Embedder:
        super().setProps(embedderType, file, dev, isHalf)

        providers = ["CPUExecutionProvider"]
        if dev.type == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.model = ort.InferenceSession(file, providers=providers)
        self.input_name = self.model.get_inputs()[0].name
        self.output_names = [o.name for o in self.model.get_outputs()]
        return self

    def _select_output_name(self, embOutputLayer: int, useFinalProj: bool) -> str:
        if embOutputLayer <= 9 and "units9" in self.output_names:
            return "units9"

        if useFinalProj and "unit12s" in self.output_names:
            return "unit12s"
        if "unit12" in self.output_names:
            return "unit12"
        if "unit12s" in self.output_names:
            return "unit12s"
        if "units9" in self.output_names:
            return "units9"

        return self.output_names[0]

    def extractFeatures(
        self, feats: torch.Tensor, embOutputLayer=9, useFinalProj=True
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("OnnxContentvec model is not loaded")

        feats_np = feats.detach().to("cpu", dtype=torch.float32).numpy()
        outputs = self.model.run(None, {self.input_name: feats_np})
        output_map = dict(zip(self.output_names, outputs))
        out_name = self._select_output_name(embOutputLayer, useFinalProj)
        selected = output_map[out_name]

        out = torch.from_numpy(np.asarray(selected)).to(self.dev)
        if self.isHalf:
            out = out.half()
        else:
            out = out.float()
        return out
