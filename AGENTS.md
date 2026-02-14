# AGENTS.md

## 適用範囲
- このファイルの指示は、リポジトリ直下（この階層）とその配下に適用する。

## 目的
- `voice-changer` リポジトリで作業するエージェント向けの運用ガイド。
- 迷ったときは「影響範囲を最小化し、再現可能な手順で変更する」を優先する。

## コミュニケーション方針
- ユーザーへの返信は日本語で行う。
- 変更前に「何を・なぜ」変更するかを短く共有する。
- 実行できなかった項目は、理由と代替案を明記する。

## リポジトリ構成（要点）
- `server/`: 音声変換サーバ本体（FastAPI + Socket.IO、推論処理）。
- `client/lib/`: クライアント向けライブラリ（TypeScript）。
- `client/demo/`: Web デモ UI（React + TypeScript）。
- `recorder/`: 録音支援ツール（React + TypeScript）。
- `docker*`, `start_*.sh`: Docker 実行関連。
- `docs/`, `docs_i18n/`, `tutorials/`: ドキュメント。

## 主要エントリーポイント
- サーバ起動: `server/MMVCServerSIO.py`
- REST 初期化: `server/restapi/MMVC_Rest.py`
- Socket.IO 初期化: `server/sio/MMVC_SocketIOServer.py`
- 推論統括: `server/voice_changer/VoiceChangerManager.py`
- Demo フロント起点: `client/demo/src/000_index.tsx`

## よく使うコマンド
- サーバ依存の導入:
  - `cd server`
  - `uv sync`
- クライアント（lib）:
  - `cd client/lib`
  - `npm install`
  - `npm run build:dev`
- クライアント（demo）:
  - `cd client/demo`
  - `npm install`
  - `npm run start`

## 変更ポリシー
- 依頼範囲外の大規模リファクタや命名変更はしない。
- 既存ファイルのスタイル・設計意図を優先する。
- 依存追加や設定変更を行う場合は、変更理由を記録する。
- 機密情報（鍵、トークン、個人情報）を追加しない。

## 作業チェックリスト
- [ ] 変更理由を説明できる
- [ ] 影響箇所を把握している
- [ ] 最低限の動作確認を実施した
- [ ] 必要なら README / コメントを更新した
