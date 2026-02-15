from torch import device

from const import EmbedderType
from voice_changer.RVC.embedder.Embedder import Embedder
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


class EmbedderManager:
    currentEmbedder: Embedder | None = None
    params: VoiceChangerParams

    @classmethod
    def initialize(cls, params: VoiceChangerParams):
        cls.params = params

    @classmethod
    def getEmbedder(cls, embederType: EmbedderType, isHalf: bool, dev: device) -> Embedder:
        if cls.currentEmbedder is None:
            print("[Voice Changer] generate new embedder. (no embedder)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, isHalf, dev)
        elif cls.currentEmbedder.matchCondition(embederType) is False:
            print("[Voice Changer] generate new embedder. (not match)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, isHalf, dev)
        else:
            print("[Voice Changer] generate new embedder. (anyway)")
            cls.currentEmbedder = cls.loadEmbedder(embederType, isHalf, dev)

            # cls.currentEmbedder.setDevice(dev)
            # cls.currentEmbedder.setHalf(isHalf)
        return cls.currentEmbedder

    @classmethod
    def loadEmbedder(cls, embederType: EmbedderType, isHalf: bool, dev: device) -> Embedder:
        if embederType == "hubert_base":
            try:
                if cls.params.content_vec_500_onnx_on is False:
                    raise Exception("[Voice Changer][Embedder] onnx is off")
                from voice_changer.RVC.embedder.OnnxContentvec import OnnxContentvec

                file = cls.params.content_vec_500_onnx
                return OnnxContentvec().loadModel(file, dev, isHalf=isHalf, embedderType="hubert_base")
            except Exception as e:  # noqa
                print("[Voice Changer] use torch contentvec", e)
                from voice_changer.RVC.embedder.FairseqHubert import FairseqHubert

                file = cls.params.hubert_base
                return FairseqHubert().loadModel(file, dev, isHalf)
        elif embederType == "hubert-base-japanese":
            from voice_changer.RVC.embedder.FairseqHubertJp import FairseqHubertJp

            file = cls.params.hubert_base_jp
            return FairseqHubertJp().loadModel(file, dev, isHalf)
        elif embederType == "contentvec":
            try:
                if cls.params.content_vec_500_onnx_on is False:
                    raise Exception("[Voice Changer][Embedder] onnx is off")
                from voice_changer.RVC.embedder.OnnxContentvec import OnnxContentvec

                file = cls.params.content_vec_500_onnx
                return OnnxContentvec().loadModel(file, dev, isHalf=isHalf, embedderType="contentvec")
            except Exception as e:
                print(e)
                from voice_changer.RVC.embedder.FairseqContentvec import FairseqContentvec

                file = cls.params.hubert_base
                return FairseqContentvec().loadModel(file, dev, isHalf)
        elif embederType == "whisper":
            from voice_changer.RVC.embedder.Whisper import Whisper

            file = cls.params.whisper_tiny
            return Whisper().loadModel(file, dev, isHalf)
        else:
            from voice_changer.RVC.embedder.FairseqHubert import FairseqHubert

            file = cls.params.hubert_base
            return FairseqHubert().loadModel(file, dev, isHalf)
