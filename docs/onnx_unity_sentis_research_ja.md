# voice-changer: ONNX変換とUnity Sentis対応調査（RVC優先）

更新日: 2026-02-14  
対象リポジトリ: `voice-changer`  
調査スコープ: RVC系モデルをUnityで動かすためのONNX変換方針とSentis制約の整理

## 1. 結論（先に要点）

- `voice-changer` の現行RVC ONNX出力は **opset 17 / int64入力前提** で、Unity Sentis向けとしてはそのままではリスクが高い。
- Unity Sentis 2.5.0（`com.unity.ai.inference`）の公式ドキュメント上は、ONNXは **opset 7-15が推奨範囲**。
- 参照実装（`uStyle-Bert-VITS2`, `piper-plus`）は、Sentis互換のために **opset 15固定** と **int64 -> int32化** を採用している。
- 初期方針は、RVC用に「Sentis向け専用エクスポート」を別系統で用意し、既存エクスポート（ORT向け）と併存するのが安全。
- Sentisで詰まるケースを想定し、ONNX Runtime（Unity）へのフォールバック案を同時に保持する。

## 2. 調査対象と確認ファイル

### 2.1 voice-changer（今回の対象）

- `server/voice_changer/RVC/onnxExporter/export2onnx.py`
- `server/voice_changer/RVC/inferencer/OnnxRVCInferencer.py`
- `server/voice_changer/RVC/pipeline/Pipeline.py`
- `server/restapi/MMVC_Rest_Fileuploader.py`

### 2.2 参照プロジェクト

- `uStyle-Bert-VITS2/scripts/convert_for_sentis.py`
- `uStyle-Bert-VITS2/scripts/convert_bert_for_sentis.py`
- `uStyle-Bert-VITS2/docs/01_onnx_export.md`
- `uStyle-Bert-VITS2/docs/03_unity_sentis_integration.md`
- `uStyle-Bert-VITS2/Packages/manifest.json`
- `piper-plus/src/python/piper_train/export_onnx.py`
- `uPiper/Assets/uPiper/Runtime/Core/AudioGeneration/InferenceAudioGenerator.cs`
- `uPiper/Packages/manifest.json`

### 2.3 Sentis一次情報（ローカルPackageCache）

- `uStyle-Bert-VITS2/Library/PackageCache/com.unity.ai.inference@e760aa121ec7/package.json`（Sentis 2.5.0）
- `.../Documentation~/supported-models.md`
- `.../Documentation~/create-an-engine.md`
- `.../Documentation~/how-sentis-runs-a-model.md`
- `.../Documentation~/supported-operators.md`
- `.../Editor/ONNX/ONNXNodeWrapper.cs`

## 3. voice-changer現状（RVC）

### 3.1 ONNX出力仕様

`server/voice_changer/RVC/onnxExporter/export2onnx.py` より:

- `torch.onnx.export(..., opset_version=17, ...)`
- 入力テンソル:
  - `feats` (float16/float32)
  - `p_len` (int64: `LongTensor`)
  - `pitch` (int64, f0あり時)
  - `pitchf` (float32, f0あり時)
  - `sid` (int64)
- `onnxsim.simplify()` を実施
- メタデータを埋め込み

### 3.2 実行時の型前提

`server/voice_changer/RVC/inferencer/OnnxRVCInferencer.py` より:

- ONNX実行時も `p_len`, `pitch`, `sid` を `np.int64` として投入。

### 3.3 パイプライン構造上の注意

`server/voice_changer/RVC/pipeline/Pipeline.py` より:

- RVC本体のONNXは「最終推論器」であり、前段に以下が存在:
  - Embedder特徴量抽出
  - F0抽出
  - 特徴量前後処理（補間・index検索等）
- つまり「ONNXをUnityに持っていくだけ」では不足で、最低限 `feats/pitch/pitchf` 供給設計が必要。

## 4. Unity Sentis制約（公式一次情報）

### 4.1 モデル互換

`supported-models.md` より:

- Sentisは ONNXを「opset 7-15」の範囲を主にサポート。
- 15を超えるモデルは import できる場合があっても結果は不定になり得る。

### 4.2 バックエンド制約

`create-an-engine.md` / `how-sentis-runs-a-model.md` より:

- `BackendType.CPU`, `GPUCompute`, `GPUPixel` を選択。
- 選択バックエンドで未対応演算があると assert になる場合がある。
- GPU非対応演算はCPUフォールバックが発生し、GPU<->CPU転送コストで性能劣化。
- `ReadbackAndClone` は同期的にブロックしうるため、頻繁な呼び出しは避ける。

### 4.3 データ型

- `tensor-fundamentals.md`: ユーザーAPIは `Tensor<float>` と `Tensor<int>` が基本。
- `NativeTensorArray.cs`: `DataType.Int` は32bit整数。
- `ONNXNodeWrapper.cs`: ONNXの `Int32/Int64/...` は importer上で `DataType.Int` にマッピングされる。

補足:

- importerはint64を受ける実装があるが、モデル全体の互換性・演算子対応・実行結果安定性の観点で、実運用では int32寄せが無難。

### 4.4 演算子対応

`supported-operators.md` より:

- オペレータごとに backend別対応差あり。
- 例:
  - `NonZero`: CPUのみ
  - `TopK`: GPUPixel非対応
  - `Shape`: CPU tensorとして扱う注記あり

## 5. 参照プロジェクトからの実践知見

### 5.1 uStyle-Bert-VITS2

- `scripts/convert_for_sentis.py`:
  - Sentis向けに int64 -> int32変換関数を用意（input/initializer/Constant/Cast/value_infoまで処理）
  - `onnxsim` と FP16変換オプション（`keep_io_types=True`）
- `scripts/convert_bert_for_sentis.py`:
  - opset 15 export
  - int64 -> int32変換
- `Packages/manifest.json`:
  - `com.unity.ai.inference: 2.5.0`

### 5.2 piper-plus

- `src/python/piper_train/export_onnx.py`:
  - `OPSET_VERSION = 15`
  - 動的軸を保ったONNX export

### 5.3 uPiper

- `InferenceAudioGenerator.cs`:
  - Sentis操作をメインスレッド側で実施
  - GPU失敗時にCPUフォールバック
  - MetalやGPUComputeでの実運用上の問題を考慮したバックエンド選択ロジック
- `Packages/manifest.json`:
  - `com.unity.ai.inference: 2.2.2`

## 6. ギャップ分析（voice-changer RVC vs Sentis）

### 6.1 互換性ギャップ

- `opset 17`（現行） vs Sentis推奨 `<=15`
- int64中心I/O前提（現行） vs Sentis実運用ではint32運用が安定

### 6.2 実装ギャップ

- Unity側に前段処理（embedder/F0）の等価実装が未定
- まずは「RVC最終推論器ONNXをSentisで動かす」PoCを切り出す必要がある

### 6.3 パフォーマンス/運用ギャップ

- バックエンド依存の差が大きい
- CPUフォールバック多発時はリアルタイム性が崩れる

## 7. 推奨方針（第1段）

### 7.1 変換方針

RVC向けに **Sentis専用ONNX生成ルート** を追加し、既存ルートは温存する。

推奨仕様:

- opset: `15`
- dtype: 整数系は `int32` へ統一（input/initializer/Constant/Cast/value_info）
- simplify: `onnxsim` 有効
- precision: 初版はFP32固定（FP16は後段で検証）

### 7.2 併存戦略

- 既存 `export2onnx.py` はORT互換維持（現行運用を壊さない）
- 新規に Sentis向けスクリプトを分離する

想定ファイル名:

- `server/voice_changer/RVC/onnxExporter/export2onnx_sentis.py`（新規）

実装状況:

- 2026-02-14 時点で `server/voice_changer/RVC/onnxExporter/export2onnx_sentis.py` を追加済み  
  （`opset=15`, `onnxsim`, `int64->int32`, CLI `--slot-index` 対応）
- 2026-02-14 時点で `server/voice_changer/RVC/onnxExporter/verify_sentis_onnx.py` を追加済み  
  （opset上限チェック、int64残存検査、ORTダミー推論）

### 7.3 検証環境

- 初期ターゲット: **Windows + DX12**
- backend優先順:
  1. `GPUCompute`
  2. `GPUPixel`
  3. `CPU`

## 8. Unity Sentis検証シナリオ（RVC最小PoC）

### 8.1 Python側

- 生成ONNXの検証:
  - opset=15
  - int64残存チェック
  - ORTでshape整合チェック

### 8.2 Unity側

- `ModelLoader.Load` 成功確認
- `Worker.Schedule` 1回完走確認
- backend切替時の挙動確認（GPU失敗時CPUフォールバック）

### 8.3 合格条件

- Unity上でRVC推論器ONNXが読み込み・実行できる
- 入出力shapeとdtypeが設計どおり
- バックエンド変更でクラッシュしない

## 9. ORT代替案（Sentisで詰まった場合）

Sentis実行において、以下が解消できない場合はONNX Runtime経路へ切替:

- opset/演算子互換問題
- GPU backendでの不安定挙動
- CPUフォールバック過多による性能不足

候補:

- `com.github.asus4.onnxruntime` 系パッケージを使ったUnity統合

## 10. 次フェーズ実装タスク（チェックリスト）

- [x] `export2onnx_sentis.py` の新規追加（opset15/int32変換対応）
- [x] 変換結果の自動検証スクリプト追加（opset/dtype/shape）
- [ ] Unity検証用の最小Runner作成（RVC推論器単体）
- [ ] backendフォールバック戦略の実装
- [ ] RVC前段処理（embedder/F0）のUnity実装方針を確定

## 11. 追加済みスクリプトの使い方（最新版）

Sentis向けに追加した `export2onnx_sentis.py` は、slot情報（`logs/<slot>/params.json`）を使って変換する。

依存導入（`server` ディレクトリ）:

```bash
uv add numpy onnx onnxsim onnxruntime torch
```

変換のみ:

```bash
uv run python -m voice_changer.RVC.onnxExporter.export2onnx_sentis \
  --slot-index 0 \
  --model-dir logs \
  --output-dir tmp_dir
```

変換 + 検証（推奨）:

```bash
uv run python -m voice_changer.RVC.onnxExporter.export2onnx_sentis \
  --slot-index 0 \
  --model-dir logs \
  --output-dir tmp_dir \
  --verify
```

生成済みONNXの単体検証:

```bash
uv run python -m voice_changer.RVC.onnxExporter.verify_sentis_onnx \
  --onnx tmp_dir/model_sentis_op15_fp32_simple.onnx \
  --max-opset 15 \
  --seq-len 64 \
  --batch-size 1
```

FP16（CUDA使用可能時のみ）:

```bash
uv run python -m voice_changer.RVC.onnxExporter.export2onnx_sentis \
  --slot-index 0 \
  --model-dir logs \
  --output-dir tmp_dir \
  --fp16 --gpu 0
```
