# eichi-plus(まだplusには程遠い版)

[FramePack-eichi](https://github.com/git-ai-code/FramePack-eichi)をベースに色々改造を施した（施す予定）プロジェクトです。

- 自分が読みやすいようにコードをリファクタリング
- 自分環境（RTX4070 12GB、RAM 32GB）に特化させた設定
  - DynamicSwapにストレージSwapを追加（ただし、これが動くと処理速度が実用的ではなくなります。ほぼ失敗作）
  - TextEncoderManagerとTransfomerManagerを以前の仕様にロールバック
  - LoRA機能の暫定削除（後ほど対応）
  - transfomerのFP8対応(ここのコードを参考にしました。[kohya-ss/FramePack-eichi](https://github.com/kohya-ss/FramePack-eichi/tree/enable-fp8-without-lora))

といった対応を行っています。  
これで、自分の環境では、1秒の動画で6分程度、4秒の動画で15分程度で生成出来ています。  
個人的には、実用範囲かなというところです。

## 開発方針
FramePack-eichiのコードのコメントから、Diffusers依存を減らす方向？のようですが、私はDiffusersにガッツリ依存してもいいと思っているので、その方向性で改造していきます。  
できるだけ、既存ライブラリで対応できるものは既存ライブラリを使う方針です。

- 予定
  - F1推論の追加
  - ループモードの拡張(参考：[red-polo/FramePackLoop](https://github.com/red-polo/FramePackLoop))
  - 動画のフレーム抽出機能の追加
  - 画像のアップスケール機能の追加
  - SDXL系をDiffusersから利用して統合、初期フレーム画像の生成や、フレームのi2iを行えるようにする。

その他、思いついたり、参考にできる機能があれば気分次第で追加します。  
要望があればissueや[X](https://x.com/siouni_unia)のDMなどに連絡いただければ、気分次第で対応したりしなかったり。

**支援のお願い**  
貧乏なのでグラボが買えません。  
余裕のある方はもしよければ、下記サイトなどからご支援いただけると嬉しいです。  
[![OFUSE](https://github.com/user-attachments/assets/62ffa121-f975-4080-923a-7f55f3cf3759)](https://ofuse.me/siouni)

[mond](https://mond.how/ja/siouni_unia)  
匿名メッセージを募集しています。  
質問などはこちらでも構いません。

[prompton](https://prompton.io/siouni_unia)  
AIイラストの有償依頼を受け付けいています。  
無償プランもあるので、お試しならそちらでも。

patreon(準備中)  
R18系のイラストをメンバーシップ向けに公開予定。

---

# FramePack-eichi | [English](README/README_en.md) | [繁體中文](README/README_zh.md) | [Русский](README/README_ru.md)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/git-ai-code/FramePack-eichi)

FramePack-eichiは、lllyasviel師の[lllyasviel/FramePack](https://github.com/lllyasviel/FramePack)のフォークであるnirvash氏の[nirvash/FramePack](https://github.com/nirvash/FramePack)を元にして作成された機能追加バージョンです。nirvash氏の先駆的な改良に基づき、細かい機能が多数搭載されています。
また、v1.9よりKohya Tech氏許諾のもと[kohya-ss/FramePack-LoRAReady](https://github.com/kohya-ss/FramePack-LoRAReady)のコードを導入し、LoRA機能の性能と安定性が大幅に向上しています。

## 📘 名称の由来

**Endframe Image CHain Interface (EICHI)**
- **E**ndframe: エンドフレーム機能の強化と最適化
- **I**mage: キーフレーム画像処理の改善と視覚的フィードバック
- **CH**ain: 複数のキーフレーム間の連携と関係性の強化
- **I**nterface: 直感的なユーザー体験とUI/UXの向上

「eichi」は日本語の「叡智」（深い知恵、英知）を連想させる言葉でもあり、AI技術の進化と人間の創造性を組み合わせるという本プロジェクトの哲学を象徴しています。
つまり叡智な差分画像から動画を作成することに特化した~~現地~~**ワールドワイド**改修仕様です。**叡智が海を越えた！**

[https://github.com/hinablue](https://github.com/hinablue) Hina Chen氏より多言語対応の協力をいただき非常に感謝しております。

## 🌟 新機能 FramePack-oichi追加 (v1.9.2)

**FramePack-eichi v1.9.2**では、新たに「FramePack-oichi」(通称：お壱)が追加されました。お壱の方は1枚の入力画像から次の1枚の未来フレーム画像を予測生成する専用ツールです。

![FramePack-oichi画面](images/framepack_oichi_screenshot.png)

### 🤔 FramePack-oichiとは？

- **1フレーム推論**: 動画全体ではなく、次の1フレームのみを生成
- **軽量処理**: 通常の動画生成よりも処理が軽く、素早く結果を確認可能
- **使いやすさ**: シンプルなUI設計で、初心者でも直感的に操作可能
- **多言語対応**: 日本語、英語、繁体字中国語に完全対応

### 💡 使用シーン

- **アイデア検討**: 画像が「次にどう動くか」を素早く確認したい時
- **リソース節約**: フル動画生成に比べて少ないリソースで処理可能
- **連続生成**: 生成した画像を入力として再度生成し、フレームを積み重ねることも可能

### 🚀 起動方法

専用の起動スクリプトが用意されています：
- 日本語版: `run_oneframe_ichi.bat`
- 英語版: `run_oneframe_ichi_en.bat`
- 繁体字中国語版: `run_oneframe_ichi_zh-tw.bat`

## 🌟 新機能 F1モデルの追加 (v1.9.1)

**FramePack-eichi v1.9.1**では、従来の逆順生成モデル「FramePack-eichi」（無印版）に加えて、順生成に対応した新モデル「FramePack-~~eichi~~ F1」が追加されました。

![FramePack-eichi画面1](images/framepack_eichi_screenshot1.png)

### 🆚 F1モデルと無印モデルの違い

| 特徴 | F1モデル | 無印モデル（無印版） |
|------|----------|------------|
| 生成方向 | 順生成（最初から最後へ） | 逆生成（最後から最初へ） |
| 動きの特徴 | 動きが多く、直感的な結果 | より精密な制御が可能 |
| UI要素 | シンプル化・簡素化 | 詳細な設定が可能 |
| 使いやすさ | 初心者向け、直感的 | 上級者向け、複雑な制御可能 |
| キーフレーム | Imageのみ | Image、Final、セクション画像 |
| 起動方法 | run_endframe_ichi_f1.bat | run_endframe_ichi.bat |

### 💡 どちらを選ぶべき？

- **初めて使う方、簡単に使いたい方** → **F1モデル**を推奨
  - より動きのある自然な結果が簡単に得られます
  - 設定項目が少なく直感的に操作できます
  - 専用の起動スクリプト（`run_endframe_ichi_f1.bat`）から利用可能

- **より高度な制御をしたい方、慣れている方** → **無印モデル**を推奨
  - 複数のキーフレーム画像を使った細かい制御が可能
  - セクションごとのプロンプト指定など高度な機能を利用可能
  - 従来の起動スクリプト（`run_endframe_ichi.bat`）から利用可能

**注意：** F1モデルの初回起動時は無印モデル版に加え、約24GBの追加モデルダウンロードが発生します。無印版のモデルもそのまま保持され、切り替えて利用できます。

## 🌟 主な機能

- **高品質な動画生成**: 単一画像から自然な動きの動画を生成　※既存機能
- **F1モデル対応**：順生成に対応した新モデルを搭載、より直感的な動画生成を実現　※v1.9.1で追加
- **FramePack-oichi**: 1枚の画像から次の1枚の未来フレーム画像を生成する新機能　※v1.9.2で追加
- **フレーム画像保存設定**: 生成した全フレーム画像または最終セクションのみの保存オプションを追加　※v1.9.2で追加
- **柔軟な動画長設定**: 1〜20秒の各セクションモードに対応　※独自機能
- **セクションフレームサイズ設定**: 0.5秒モードと1秒モードを切り替え可能　※v1.5で追加
- **オールパディング機能**: すべてのセクションで同じパディング値を使用　※v1.4で追加
- **マルチセクション対応**: 複数のセクションでキーフレーム画像を指定し複雑なアニメーションを実現　※nirvash氏追加機能
- **セクションごとのプロンプト設定**: 各セクションに個別のプロンプトを指定可能　※v1.2で追加
- **赤枠/青枠によるキーフレーム画像効率的コピー**: 2つのセクションだけで全セクションをカバー可能に　※v1.7で追加
- **テンソルデータの保存と結合**: 動画の潜在表現を保存し、複数の動画を結合可能に　※v1.8で追加
- **プロンプト管理機能**: プロンプトの保存、編集、再利用が簡単　※v1.3で追加
- **PNGメタデータ埋め込み**: 生成画像にプロンプト、シード値、セクション情報を自動的に記録　※v1.9.1で追加
- **Hunyuan/FramePack LoRAサポート**: モデルのカスタマイズによる独自の表現を追加　※v1.9/v1.9.1で大幅改善
- **LoRA機能の強化**: 3つのLoRAの同時使用に対応、/webui/loraフォルダからの選択機能　※v1.9.2で追加
- **セクション情報の一括管理**: ZIPファイルによるセクション情報のダウンロード/アップロードと内容変更後の再ダウンロードに対応　※v1.9.2で追加
- **VAEキャッシュ機能**: フレーム単位のデコードによる処理速度向上（オプション）　※v1.9.2で追加、[furusu氏の検証](https://note.com/gcem156/n/nb93535d80c82)と[FramePack実装](https://github.com/laksjdjf/FramePack)に基づく
- **FP8最適化**: LoRA適用時のVRAM使用量削減と処理速度の最適化　※v1.9.1で追加
- **MP4圧縮設定**: 動画のファイルサイズと品質のバランスを調整可能　※v1.6.2で本家からマージ
- **出力フォルダ管理機能**: 出力先フォルダの指定とOSに依存しない開き方をサポート ※v1.2で追加
- **多言語対応（i18n）**: 日本語、英語、繁体字中国語のUIをサポート　※v1.8.1で追加
- **Docker対応**: コンテナ化された環境でFramePack-eichiを簡単に実行　※v1.9.1で追加

![FramePack-eichi画面2](images/framepack_eichi_screenshot2.png)

**セクション設定使用時**
![FramePack-eichi画面3](images/framepack_eichi_screenshot3.png)

## 📚 関連資料

- **[セットアップガイド](README/README_setup.md)** - インストール方法の詳細
- **[使用ガイド](README/README_userguide.md)** - 詳細な使用方法
- **[設定情報](README/README_column.md#--設定情報)** - 設定に関する詳細情報
- **[更新履歴](README/README_changelog.md)** - 全バージョンの更新内容

## 📝 最新アップデート情報 (v1.9.2)

### 主要な変更点

#### 1. 「FramePack-oichi」新機能追加
- **1フレーム推論**: 1枚の入力画像から次の1枚の未来フレーム画像を予測生成する新機能
- **専用起動スクリプト**: `run_oneframe_ichi.bat`など多言語対応のスクリプトを追加
- **軽量処理**: 通常の動画生成より軽量で素早く結果を確認可能
- **多言語対応**: 日本語、英語、繁体字中国語に完全対応

#### 2. フレーム画像保存機能
- **全フレーム保存オプション**: 生成した全フレーム画像を保存するオプションを追加
- **選択式保存**: 全セクションのフレーム保存か最終セクションのみの保存か選択可能
- **素材活用**: 動画の中間過程の可視化や素材としての活用が容易に

#### 3. セクション情報の一括管理機能
- **ZIPによる一括管理**: セクション情報のZIPファイルによる一括ダウンロード機能を追加
- **プロジェクト管理**: 複数プロジェクトの効率的な管理・バックアップが可能に
- **一括アップロード対応**: 開始画像、終了画像、セクション情報の一括アップロードおよび変更後の再ダウンロードに対応

#### 4. LoRA機能の強化
- **複数LoRA対応**: 3つのLoRAの同時使用に対応
- **改善されたLoRA選択**: ディレクトリからの選択をデフォルトとし、/webui/loraフォルダからも選択可能
- **全モードサポート**: 無印版、F1版、oneframe版でLoRA機能強化を対応

#### 5. VAEキャッシュ機能
- **処理速度向上**: フレーム単位のVAEデコードによる処理速度の最適化
- **柔軟な設定**: メモリ使用量と処理速度のバランスを調整可能
- **簡単切替**: 設定画面から簡単にオン/オフを切り替え可能（デフォルトはOFF）
- **高速化効果**: フレーム間の独立性を活かした計算キャッシュにより最大30%程度の高速化を実現

## 📝 アップデート情報 (v1.9.1)

### 主要な変更点

#### 1. F1モデルの追加
- **順生成方式の新モデル**: 通常の生成方向（最初から最後へ）に対応したモデル「FramePack_F1_I2V_HY_20250503」を導入
- **シンプル化されたインターフェース**: F1モデルではセクション（キーフレーム画像）とFinal（endframe）機能を削除
- **専用起動スクリプト**: `run_endframe_ichi_f1.bat`および多言語版スクリプトを追加
- **Image影響度の調整機能**: F1では初回セクションのImageからの変化を抑える機能を追加（100.0%～102.0%の範囲で設定可能）

#### 2. メモリ管理の最適化
- **モデル管理機能の強化**: `transformer_manager.py`と`text_encoder_manager.py`による効率的なメモリ管理
- **FP8最適化機能**: 8ビット浮動小数点形式によるLoRA適用時のVRAM使用量削減
- **RTX 40シリーズGPU向け最適化**: `scaled_mm`最適化によるパフォーマンス向上

#### 3. PNGメタデータ機能
- **メタデータ埋め込み**: 生成画像にプロンプト、シード、セクション情報を自動的に保存
- **メタデータ抽出機能**: 保存された画像から設定を再取得可能
- **SD系との互換性**: 標準的なメタデータ形式による他ツールとの互換性確保

#### 4. クリップボード対応の拡充
- **無印版と共通化**: 無印版のImageおよびFinalとF1版のImageのクリップボード対応

#### 5. 複数セクション情報の一括追加機能
- **zipファイルインポート機能**: セクション画像とプロンプトをzipファイルで一括設定可能
- **自動セクション設定**: zipファイル内の番号付き画像とYAML設定ファイルに基づき自動設定
- **構成サポート**: 開始フレーム、終了フレーム、各セクション画像とプロンプトを一括登録

#### 6. Docker対応の強化
- **コンテナ化環境**: Dockerfile、docker-compose.ymlを使用した簡単なセットアップ
- **多言語サポート**: 複数言語（日本語、英語、中国語）に対応したコンテナイメージ

## 📝 アップデート情報 (v1.9)

### 主要な変更点

#### 1. kohya-ss/FramePack-LoRAReadyの導入
- **LoRA機能の大幅な性能向上**: Kohya Tech氏許諾の下、LoRA適用の安定性と一貫性が向上
- **高VRAMモードと低VRAMモードの統一**: どちらのモードでも同じ直接適用方式を採用
- **コード複雑性の軽減**: 共通の`load_and_apply_lora`関数使用による保守性向上
- **DynamicSwapフック方式の廃止**: より安定した直接適用方式への完全移行

#### 2. HunyuanVideo形式への統一
- **LoRAフォーマットの標準化**: HunyuanVideo形式に統一し、異なるフォーマットの互換性を向上

## 💻 インストール方法

### 前提条件

- Windows 10/11（Linux/Macでも基本機能は多分動作可能）
- NVIDIA GPU (RTX 30/40シリーズ推奨、最低8GB VRAM)
- CUDA Toolkit 12.6
- Python 3.10.x
- 最新のNVIDIA GPU ドライバー

※ Linuxでの動作はv1.2で強化され、オープン機能も追加されましたが、一部機能に制限がある場合があります。

### 手順

#### 公式パッケージのインストール

まず、元のFramePackをインストールする必要があります。

1. [公式FramePack](https://github.com/lllyasviel/FramePack?tab=readme-ov-file#installation)からWindowsワンインストーラーをダウンロードします。
   「Click Here to Download One-Click Package (CUDA 12.6 + Pytorch 2.6)」をクリックします。

2. ダウンロードしたパッケージを解凍し、`update.bat`を実行してから`run.bat`で起動します。
   `update.bat`の実行は重要です。これを行わないと、潜在的なバグが修正されていない以前のバージョンを使用することになります。

3. 初回起動時に必要なモデルが自動的にダウンロードされます（約30GB）。
   既にダウンロード済みのモデルがある場合は、`framepack\webui\hf_download`フォルダに配置してください。

4. この時点で動作しますが、高速化ライブラリ（Xformers、Flash Attn、Sage Attn）が未インストールの場合、処理が遅くなります。
   ```
   Currently enabled native sdp backends: ['flash', 'math', 'mem_efficient', 'cudnn']
   Xformers is not installed!
   Flash Attn is not installed!
   Sage Attn is not installed!
   ```
   
   処理時間の違い: ※RAM:32GB、RXT4060Ti(16GB)の場合
   - ライブラリ未インストール時: 約4分46秒/25ステップ
   - ライブラリインストール時: 約3分17秒〜3分25秒/25ステップ

5. 高速化ライブラリをインストールするには、[Issue #138](https://github.com/lllyasviel/FramePack/issues/138)から`package_installer.zip`をダウンロードし、解凍してルートディレクトリで`package_installer.bat`を実行します（コマンドプロンプト内でEnterを押す）。

6. 再度起動してライブラリがインストールされたことを確認します:
   ```
   Currently enabled native sdp backends: ['flash', 'math', 'mem_efficient', 'cudnn']
   Xformers is installed!
   Flash Attn is not installed!
   Sage Attn is installed!
   ```
   作者が実行した場合、Flash Attnはインストールされませんでした。
   注: Flash Attnがインストールされていなくても、処理速度にはほとんど影響がありません。テスト結果によると、Flash Attnの有無による速度差はわずかで、「Flash Attn is not installed!」の状態でも約3分17秒/25ステップと、すべてインストールされている場合（約3分25秒/25ステップ）とほぼ同等の処理速度を維持できます。
   Xformersが入っているかどうかが一番影響が大きいと思います。

#### FramePack-eichiのインストール

1. 実行ファイルをFramePackのルートディレクトリに配置します：
   - `run_endframe_ichi.bat` - 無印版/日本語用（デフォルト）
   - `run_endframe_ichi_en.bat` - 無印版/英語用
   - `run_endframe_ichi_zh-tw.bat` - 無印版/繁体字中国語用
   - `run_endframe_ichi_f1.bat` - F1版/日本語用 ※v1.9.1で追加
   - `run_endframe_ichi_en_f1.bat` - F1版/英語用 ※v1.9.1で追加
   - `run_endframe_ichi_zh-tw_f1.bat` - F1版/繁体字中国語用 ※v1.9.1で追加
   - `run_oneframe_ichi.bat` - 1フレーム推論/日本語用 ※v1.9.2で追加
   - `run_oneframe_ichi_en.bat` - 1フレーム推論/英語用 ※v1.9.2で追加
   - `run_oneframe_ichi_zh-tw.bat` - 1フレーム推論/繁体字中国語用 ※v1.9.2で追加

2. 以下のファイルとフォルダを`webui`フォルダに配置します：
   - `endframe_ichi.py` - 無印版メインアプリケーションファイル
   - `endframe_ichi_f1.py` - F1版メインアプリケーションファイル ※v1.9.1で追加
   - `oneframe_ichi.py` - 1フレーム推論版メインアプリケーションファイル ※v1.9.2で追加
   - `eichi_utils` フォルダ - ユーティリティモジュール
     - `__init__.py`
     - `frame_calculator.py` - フレームサイズ計算モジュール
     - `keyframe_handler.py` - キーフレーム処理モジュール
     - `keyframe_handler_extended.py` - キーフレーム処理モジュール
     - `preset_manager.py` - プリセット管理モジュール
     - `settings_manager.py` - 設定管理モジュール
     - `tensor_combiner.py` - テンソル結合モジュール ※v1.8で追加
     - `ui_styles.py` - UIスタイル定義モジュール ※v1.6.2で追加
     - `video_mode_settings.py` - 動画モード設定モジュール
     - `png_metadata.py` - PNGメタデータモジュール ※v1.9.1で追加
     - `text_encoder_manager.py` - テキストエンコーダー管理モジュール ※v1.9.1で追加
     - `transformer_manager.py` - トランスフォーマーモデル管理モジュール ※v1.9.1で追加
     - `section_manager.py` - セクション情報管理モジュール ※v1.9.2で追加
     - `vae_cache.py` - VAEキャッシュモジュール ※v1.9.2で追加
   - `lora_utils` フォルダ - LoRA関連モジュール
     - `__init__.py`
     - `dynamic_swap_lora.py` - LoRA管理モジュール（後方互換性用に維持）
     - `lora_loader.py` - LoRAローダーモジュール
     - `lora_check_helper.py` - LoRA適用状況確認モジュール
     - `lora_utils.py` - LoRA状態辞書マージや変換機能 ※v1.9で追加
     - `fp8_optimization_utils.py` - FP8最適化機能 ※v1.9.1で追加
   - `diffusers_helper` フォルダ - モデルのメモリ管理改善用ユーティリティ ※v1.9で追加
     - `memory.py` - メモリ管理機能を提供
     - `bucket_tools.py` - 解像度バケット機能 ※v1.9.1で追加
     - **注意**: このディレクトリは既存の本家ツールのソースを差し替えるため、必要に応じてバックアップを取ってください
   - `locales` フォルダ - 多言語対応モジュール ※v1.8.1で追加
     - `i18n.py` - 国際化（i18n）機能のコア実装
     - `i18n_extended.py` - 国際化機能の拡張 ※v1.9.1で追加
     - `ja.json` - 日本語の翻訳ファイル（デフォルト言語）
     - `en.json` - 英語の翻訳ファイル
     - `zh-tw.json` - 繁体字中国語の翻訳ファイル

3. 希望するバージョンと言語の実行ファイルを実行すると、対応するWebUIが起動します：
   - 無印版/日本語：`run_endframe_ichi.bat`
   - 無印版/英語：`run_endframe_ichi_en.bat`
   - 無印版/繁体字中国語：`run_endframe_ichi_zh-tw.bat`
   - F1版/日本語：`run_endframe_ichi_f1.bat` ※v1.9.1で追加
   - F1版/英語：`run_endframe_ichi_en_f1.bat` ※v1.9.1で追加
   - F1版/繁体字中国語：`run_endframe_ichi_zh-tw_f1.bat` ※v1.9.1で追加
   - 1フレーム推論版/日本語：`run_oneframe_ichi.bat` ※v1.9.2で追加
   - 1フレーム推論版/英語：`run_oneframe_ichi_en.bat` ※v1.9.2で追加
   - 1フレーム推論版/繁体字中国語：`run_oneframe_ichi_zh-tw.bat` ※v1.9.2で追加

   または、コマンドラインから直接言語を指定して起動することも可能です：
   ```bash
   python endframe_ichi.py --lang en  # 無印版/英語で起動
   python endframe_ichi_f1.py --lang zh-tw  # F1版/繁体字中国語で起動
   python oneframe_ichi.py --lang en  # 1フレーム推論版/英語で起動
   ```

#### Docker インストール
FramePack-eichiはDockerを使用して簡単にセットアップでき、異なるシステム間で一貫した環境を提供します。

##### Docker インストールの前提条件
- システムにDockerがインストールされていること
- システムにDocker Composeがインストールされていること
- NVIDIA GPU（最低8GB VRAM、RTX 30/40シリーズ推奨）

##### Dockerセットアップ手順
1. **言語選択**:
   Dockerコンテナはデフォルトで英語で起動するように設定されています。`docker-compose.yml`の`command`パラメータを変更することで、これを変更できます：
   ```yaml
   # 日本語の場合:
   command: ["--lang", "ja"]
   
   # 繁体字中国語の場合:
   command: ["--lang", "zh-tw"]
   
   # 英語の場合（デフォルト）:
   command: ["--lang", "en"]
   ```

2. **コンテナのビルドと起動**:
   ```bash
   # コンテナをビルド（初回またはDockerfile変更後）
   docker-compose build
   
   # コンテナを起動
   docker-compose up
   ```
   
   バックグラウンドで実行する場合（デタッチモード）:
   ```bash
   docker-compose up -d
   ```

3. **Web UIへのアクセス**:
   コンテナが実行されると、次のURLでWebインターフェースにアクセスできます:
   ```
   http://localhost:7861
   ```

4. **初回実行の注意点**:
   - 初回実行時、コンテナは必要なモデル（約30GB）をダウンロードします
   - F1版を使用する場合は追加で約24GBのダウンロードが発生します
   - 初期起動時に「h11エラー」が表示される場合があります（トラブルシューティングセクションを参照）
   - すでにモデルをダウンロードしている場合は、`./models`ディレクトリに配置してください

#### Linux向けインストール方法

Linuxでは、以下の手順で実行可能です：

1. 上記の必要なファイルとフォルダをダウンロードして配置します。
2. ターミナルで次のコマンドを実行します：
   ```bash
   python endframe_ichi.py  # 無印版
   # または
   python endframe_ichi_f1.py  # F1版
   ```

#### Google Colab向けインストール方法

- [Google Colab で FramePack を試す](https://note.com/npaka/n/n33d1a0f1bbc1)をご参考ください

#### Mac(mini M4 Pro)向けインストール方法

- [話題のFramePackをMac mini M4 Proで動かしてみた件](https://note.com/akira_kano24/n/n49651dbef319)をご参考ください

## 🚀 使い方

### モデル選択

1. **使用目的に合わせたモデル選択**:
   - **初めての方、簡単に使いたい方**: F1モデル（`run_endframe_ichi_f1.bat`）を使用
   - **細かい制御をしたい経験者**: 無印モデル（`run_endframe_ichi.bat`）を使用

2. **モデルの特徴**:
   - **F1モデル**: 順生成方式、よりダイナミックな動き、簡単な操作
   - **無印モデル**: 逆生成方式、精密な制御、多機能

### 基本的な動画生成

1. **画像をアップロード**: 「Image」枠に画像をアップロード
2. **プロンプトを入力**: キャラクターの動きを表現するプロンプトを入力
3. **設定を調整**: 動画長やシード値を設定
4. **生成開始**: 「Start Generation」ボタンをクリック

### F1モデル特有の設定 (v1.9.1)

- **Image影響度**: 初回セクションでの変化度合いを調整（100.0%～102.0%）
- **プロンプト**: 動きの表現に重点を置いたプロンプトが効果的
- **注意**: F1モデルではセクション設定とFinal Frameは使用できません

### 無印モデルの高度な設定

- **生成モード選択**:
  - **通常モード**: 一般的な動画生成
  - **ループモード**: 最終フレームが最初のフレームに戻る循環動画を生成

- **オールパディング選択**:
  - **オールパディング**: すべてのセクションで同じパディング値を使用
  - **パディング値**: 0〜3の整数値

- **動画長設定**:
  - **1～20秒**

[続きはこちら](README/README_column.md#--高度な設定)

## 🛠️ 設定情報

### FP8最適化設定 (v1.9.1)

- **FP8最適化**: 8ビット浮動小数点形式によるLoRA適用時のVRAM使用量削減
- **RTX 40シリーズGPU**: scaled_mm最適化による高速化
- **通常は無効**: 有効にすると低VRAM環境でもLoRAが使いやすくなります

### 言語設定

1. **実行ファイルによる言語選択**:
   - `run_endframe_ichi.bat` - 無印版/日本語（デフォルト）
   - `run_endframe_ichi_en.bat` - 無印版/英語
   - `run_endframe_ichi_zh-tw.bat` - 無印版/繁体字中国語
   - `run_endframe_ichi_f1.bat` - F1版/日本語
   - `run_endframe_ichi_en_f1.bat` - F1版/英語
   - `run_endframe_ichi_zh-tw_f1.bat` - F1版/繁体字中国語

2. **コマンドラインによる言語指定**:
   ```
   python endframe_ichi.py --lang en  # 英語で起動
   python endframe_ichi_f1.py --lang zh-tw  # F1版/繁体字中国語で起動
   ```

※ READMEの多言語版も順次対応予定です。繁体字中国語版は[README/README_zh.md](README/README_zh.md)をご参照ください。

## 🔧 トラブルシューティング

### h11エラーについて

ツールを立ち上げ、初回に画像をインポートする際に以下のようなエラーが多数発生することがあります：
※コンソールにエラーが表示され、GUI上では画像が表示されません。

**実際には画像はアップロードされており、サムネイルの表示に失敗しているケースが大半のため、そのまま動画生成いただくことも可能です。**

![FramePack-eichiエラー画面1](images/framepack_eichi_error_screenshot1.png)
```
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "C:\xxx\xxx\framepack\system\python\lib\site-packages\uvicorn\protocols\http\h11_impl.py", line 404, in run_asgi
```
このエラーは、HTTPレスポンスの処理中に問題が発生した場合に表示されます。
前述のとおり、Gradioが起動し終わっていない起動初期の段階で頻発します。

解決方法：
1. 画像を一度「×」ボタンで削除して、再度アップロードしてみてください。
2. 同じファイルで何度も失敗する場合：
   - Pythonプロセスを完全に停止してからアプリケーションを再起動
   - PCを再起動してからアプリケーションを再起動

エラーが継続する場合は、他の画像ファイルを試すか、画像サイズを縮小してみてください。

### メモリ不足エラー

「CUDA out of memory」や「RuntimeError: CUDA error」が表示される場合：

1. `gpu_memory_preservation` の値を大きくする（例: 12-16GB）
2. 他のGPUを使用するアプリケーションを閉じる
3. 再起動して再試行
4. 画像解像度を下げる（640x640前後推奨）

### メモリ消費量の詳細分析

ソースコード全体の詳細分析に基づく、実際のメモリ消費量は以下の通りです：

#### 基本モデル構成とロード時のメモリ消費

- **transformer** (FramePackI2V_HY): 約25-30GB (状態辞書の完全展開時)
- **text_encoder**, **text_encoder_2**, **vae** 合計: 約8-10GB
  ｖ1.9.1ではアンロード処理を追加し、RAM使用量を削減しています
- **その他補助モデル** (image_encoder等): 約3-5GB
- **合計ベースラインRAM消費**: 約36-45GB

#### LoRA適用時のメモリ消費パターン

LoRA適用時の処理フロー：
1. **元の状態辞書**: 約25-30GB (transformerのサイズ)
2. **LoRAファイル読み込み**: 50MB〜500MB
3. **マージ処理時**: 元の状態辞書を複製しつつLoRAと結合するため、**一時的に追加で25-30GB必要**
　　※v1.9.1では処理を見直し、RAM消費を大幅に削減しています
4. **適用処理ピーク時**: 基本消費量 + 状態辞書複製 ≈ 約70-90GB

**重要**: この大幅なメモリ消費増加は初回のLoRAロード時のみ発生します。別のLoRAファイルを読み込む場合や、LoRAの適用強度、FP8の設定変更時以外は、このような大きなメモリピークは発生しません。通常の動画生成操作では、初回ロード後のメモリ使用量は基本的な消費量（36-45GB程度）に留まります。

#### 推奨システム要件

- **標準実行時必要RAM**: 36-45GB
- **LoRA適用時の一時ピーク**: 70-90GB
- **推奨ページングファイル**: RAMを合わせて40GB以上確保が望ましい
  
例えば:
- 32GB RAMシステム → ページングファイル 40GB程度
- 64GB RAMシステム → ページングファイル 20GB程度

#### F1モデル使用時の追加メモリ要件

F1モデルを使用する場合、無印モデルに加えて以下の追加メモリが必要です：
- **F1モデル初回ダウンロード**: 約24GB
- **両モデル共存時の保存容量**: 約54GB

#### Google Colabなどのスワップなし環境での注意

スワップメモリのない環境での実行は非常に厳しい制約があります:
- RAM制限 (13-15GB) では基本モデルのロードも困難
- LoRA適用時に必要な70-90GBの一時メモリピークに対応できない
- スワップがないため、メモリ不足時は即座にOOMエラーでクラッシュする

### 動画表示の問題

一部のブラウザ（特にFirefoxなど）やmacOSで生成された動画が表示されない問題があります：

- 症状: Gradio UI上で動画が表示されない、Windowsでサムネイルが表示されない、一部のプレイヤーで再生できない
- 原因: `\framepack\webui\diffusers_helper\utils.py`内のビデオコーデック設定の問題

**こちらについては本家のMP4 Compression機能をマージし解消しました**

## 🤝 謝辞

このプロジェクトは以下のプロジェクトの貢献に基づいています：

- [lllyasviel/FramePack](https://github.com/lllyasviel/FramePack) - 原作者lllyasviel師の素晴らしい技術と革新性に感謝します
- [nirvash/FramePack](https://github.com/nirvash/FramePack) - nirvash氏の先駆的な改良と拡張に深く感謝します
- [kohya-ss/FramePack-LoRAReady](https://github.com/kohya-ss/FramePack-LoRAReady) - kohya-ss氏のLoRA対応コードの提供により、v1.9での性能向上が実現しました

## 📄 ライセンス

本プロジェクトは[Apache License 2.0](LICENSE)の下で公開されています。これは元のFramePackプロジェクトのライセンスに準拠しています。


## 📝 更新履歴

最新の更新情報を以下に示します。更新履歴の全文は[更新履歴](README/README_changelog.md)をご参照ください。

### 2025-05-11: バージョン1.9.2
- **「FramePack-oichi」新機能追加**: 1枚の入力画像から次の1枚の未来フレーム画像を予測生成
- **フレーム画像保存機能**: 生成した全フレーム画像を保存するオプションを追加
- **セクション情報の一括管理機能**: ZIPファイルによるセクション情報のダウンロード・アップロード対応
- **LoRA機能の強化**: 3つのLoRAの同時使用に対応、/webui/loraフォルダからの選択機能
- **VAEキャッシュ機能**: フレーム単位でのVAEデコードによる処理速度向上（オプション機能）※furusu氏の検証に基づく実装 [詳細1](https://note.com/gcem156/n/nb93535d80c82) [詳細2](https://github.com/laksjdjf/FramePack)

### 2025-05-04: バージョン1.9.1
- **F1モデルの追加**: 順生成に対応した新モデル「FramePack_F1_I2V_HY_20250503」を導入
- **メモリ管理機能の強化**: transformer_manager.pyとtext_encoder_manager.pyによる効率的なモデル管理
- **Docker対応の強化**: コンテナ化環境での簡単なセットアップをサポート
- **PNGメタデータ機能**: 生成画像へのプロンプト、シード、セクション情報の自動埋め込み
- **最適解像度バケットシステム**: アスペクト比に応じた最適な解像度を自動選択
- **クリップボード対応の拡充**: 無印版のImageおよびFinalとF1版のImageのクリップボード対応

---
**FramePack-eichi** - Endframe Image CHain Interface  
より直感的で、より柔軟な動画生成を目指して