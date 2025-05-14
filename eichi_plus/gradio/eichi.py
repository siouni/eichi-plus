import os

import random
import gradio as gr
import traceback  # デバッグログ出力用
import zipfile
import yaml
import torch
import argparse

import eichi_plus.shared.eichi as eichi_shared
import eichi_plus.shared.eichi_plus as eichi_plus_shared
# from eichi_plus.worker.eichi import worker
from eichi_plus.worker.eichi_plus import worker

from locales.i18n_extended import translate

from eichi_utils.png_metadata import (
    extract_metadata_from_png, PROMPT_KEY, SEED_KEY, SECTION_PROMPT_KEY
)
from eichi_utils.video_mode_settings import (
    VIDEO_MODE_SETTINGS, get_video_modes, get_video_seconds, get_max_keyframes_count, MODE_TYPE_NORMAL, MODE_TYPE_LOOP
)
from eichi_utils.ui_styles import get_app_css
# プリセット管理モジュールをインポート
from eichi_utils.preset_manager import (
    load_presets,
    get_default_startup_prompt,
    save_preset,
    delete_preset
)
# 拡張キーフレーム処理モジュールをインポート
from eichi_utils.keyframe_handler_extended import extended_mode_length_change_handler
# 設定管理モジュールをインポート
from eichi_utils.settings_manager import (
    get_output_folder_path,
    load_settings,
    save_settings,
    open_output_folder
)

from diffusers_helper.gradio.progress_bar import make_progress_bar_html
from diffusers_helper.thread_utils import AsyncStream, async_run

# enable_keyframe_copyの初期化（グローバル変数）
enable_keyframe_copy = True

mode_radio = None

# 画像のバリデーション関数
def validate_images(input_image, section_settings, length_radio=None, frame_size_radio=None):
    """入力画像または画面に表示されている最後のキーフレーム画像のいずれかが有効かを確認する"""
    # 入力画像をチェック
    if input_image is not None:
        return True, ""

    # 現在の設定から表示すべきセクション数を計算
    total_display_sections = None
    if length_radio is not None and frame_size_radio is not None:
        try:
            # 動画長を秒数で取得
            seconds = get_video_seconds(length_radio.value)

            # フレームサイズ設定からlatent_window_sizeを計算
            latent_window_size = 4.5 if frame_size_radio.value == translate("0.5秒 (17フレーム)") else 9
            frame_count = latent_window_size * 4 - 3

            # セクション数を計算
            total_frames = int(seconds * 30)
            total_display_sections = int(max(round(total_frames / frame_count), 1))
            print(translate("[DEBUG] 現在の設定によるセクション数: {0}").format(total_display_sections))
        except Exception as e:
            print(translate("[ERROR] セクション数計算エラー: {0}").format(e))

    # 入力画像がない場合、表示されているセクションの中で最後のキーフレーム画像をチェック
    last_visible_section_image = None
    last_visible_section_num = -1

    if section_settings is not None and not isinstance(section_settings, bool):
        # 有効なセクション番号を収集
        valid_sections = []
        try:
            for section in section_settings:
                if section and len(section) > 1 and section[0] is not None:
                    try:
                        section_num = int(section[0])
                        # 表示セクション数が計算されていれば、それ以下のセクションのみ追加
                        if total_display_sections is None or section_num < total_display_sections:
                            valid_sections.append((section_num, section[1]))
                    except (ValueError, TypeError):
                        continue
        except (TypeError, ValueError):
            # section_settingsがイテラブルでない場合（ブール値など）、空のリストとして扱う
            valid_sections = []
            print(f"[DEBUG] section_settings is not iterable: {type(section_settings)}")


        # 有効なセクションがあれば、最大の番号（最後のセクション）を探す
        if valid_sections:
            # 番号でソート
            valid_sections.sort(key=lambda x: x[0])
            # 最後のセクションを取得
            last_visible_section_num, last_visible_section_image = valid_sections[-1]

            print(translate("[DEBUG] 最後のキーフレーム確認: セクション{0} (画像あり: {1})").format(last_visible_section_num, last_visible_section_image is not None))

    # 最後のキーフレーム画像があればOK
    if last_visible_section_image is not None:
        return True, ""

    # どちらの画像もない場合はエラー
    error_html = f"""
    <div style="padding: 15px; border-radius: 10px; background-color: #ffebee; border: 1px solid #f44336; margin: 10px 0;">
        <h3 style="color: #d32f2f; margin: 0 0 10px 0;">{translate('❗️ 画像が選択されていません')}</h3>
        <p>{translate('生成を開始する前に「Image」欄または表示されている最後のキーフレーム画像に画像をアップロードしてください。これはあまねく叡智の始発点となる重要な画像です。')}</p>
    </div>
    """
    error_bar = make_progress_bar_html(100, translate('画像がありません'))
    return False, error_html + error_bar

def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_random_seed, mp4_crf=16, all_padding_value=1.0, end_frame=None, end_frame_strength=1.0, frame_size_setting="1秒 (33フレーム)", keep_section_videos=False, lora_files=None, lora_files2=None, lora_scales_text="0.8,0.8", output_dir=None, save_section_frames=False, section_settings=None, use_all_padding=False, use_lora=False, save_tensor_data=False, tensor_data_input=None, fp8_optimization=False, resolution=640, batch_count=1):
    eichi_shared.frame_size_setting = frame_size_setting

    # バッチ処理開始時に停止フラグをリセット
    eichi_shared.batch_stopped = False

    # バリデーション関数で既にチェック済みなので、ここでの再チェックは不要

    # フレームサイズ設定に応じてlatent_window_sizeを先に調整
    if frame_size_setting == "0.5秒 (17フレーム)":
        # 0.5秒の場合はlatent_window_size=4.5に設定（実際には4.5*4-3=17フレーム≒0.5秒@30fps）
        latent_window_size = 4.5
        print(translate('フレームサイズを0.5秒モードに設定: latent_window_size = {0}').format(latent_window_size))
    else:
        # デフォルトの1秒モードではlatent_window_size=9を使用（9*4-3=33フレーム≒1秒@30fps）
        latent_window_size = 9
        print(translate('フレームサイズを1秒モードに設定: latent_window_size = {0}').format(latent_window_size))

    # バッチ処理回数を確認し、詳細を出力
    batch_count = max(1, min(int(batch_count), 100))  # 1〜100の間に制限
    print(translate("\u25c6 バッチ処理回数: {0}回").format(batch_count))

    # 解像度を安全な値に丸めてログ表示
    from diffusers_helper.bucket_tools import SAFE_RESOLUTIONS

    # 解像度値を表示
    print(translate("\u25c6 UIから受け取った解像度値: {0}（型: {1}）").format(resolution, type(resolution).__name__))

    # 安全な値に丸める
    if resolution not in SAFE_RESOLUTIONS:
        closest_resolution = min(SAFE_RESOLUTIONS, key=lambda x: abs(x - resolution))
        print(translate('安全な解像度値ではないため、{0}から{1}に自動調整しました').format(resolution, closest_resolution))
        resolution = closest_resolution

    # 解像度設定を出力
    print(translate('解像度を設定: {0}').format(resolution))

    # 動画生成の設定情報をログに出力
    # 4.5の場合は5として計算するための特別処理
    if latent_window_size == 4.5:
        frame_count = 17  # 5 * 4 - 3 = 17
    else:
        frame_count = int(latent_window_size * 4 - 3)
    total_latent_sections = int(max(round((total_second_length * 30) / frame_count), 1))

    global mode_radio
    mode_name = translate("通常モード") if mode_radio.value == MODE_TYPE_NORMAL else translate("ループモード")

    print(translate("\n==== 動画生成開始 ====="))
    print(translate("\u25c6 生成モード: {0}").format(mode_name))
    print(translate("\u25c6 動画長: {0}秒").format(total_second_length))
    print(translate("\u25c6 フレームサイズ: {0}").format(frame_size_setting))
    print(translate("\u25c6 生成セクション数: {0}回").format(total_latent_sections))
    print(translate("\u25c6 サンプリングステップ数: {0}").format(steps))
    print(translate("\u25c6 TeaCache使用: {0}").format(use_teacache))
    # TeaCache使用の直後にSEED値の情報を表示
    print(translate("\u25c6 使用SEED値: {0}").format(seed))
    print(translate("\u25c6 LoRA使用: {0}").format(use_lora))

    # FP8最適化設定のログ出力
    print(translate("\u25c6 FP8最適化: {0}").format(fp8_optimization))

    # オールパディング設定のログ出力
    if use_all_padding:
        print(translate("\u25c6 オールパディング: 有効 (値: {0})").format(round(all_padding_value, 1)))
    else:
        print(translate("\u25c6 オールパディング: 無効"))

    # LoRA情報のログ出力
    if use_lora and eichi_shared.has_lora_support:
        all_lora_files = []
        
        # 1つ目のLoRAファイルを処理
        if lora_files is not None:
            if isinstance(lora_files, list):
                all_lora_files.extend(lora_files)
            else:
                all_lora_files.append(lora_files)
                
        # 2つ目のLoRAファイルを処理
        if lora_files2 is not None:
            if isinstance(lora_files2, list):
                all_lora_files.extend(lora_files2)
            else:
                all_lora_files.append(lora_files2)
        
        # スケール値を解析
        try:
            scales = [float(s.strip()) for s in lora_scales_text.split(',')]
        except:
            # 解析エラーの場合はデフォルト値を使用
            scales = [0.8] * len(all_lora_files)
            
        # スケール値の数を調整
        if len(scales) < len(all_lora_files):
            scales.extend([0.8] * (len(all_lora_files) - len(scales)))
        elif len(scales) > len(all_lora_files):
            scales = scales[:len(all_lora_files)]
            
        # LoRAファイル情報を出力
        if len(all_lora_files) == 1:
            # 単一ファイル
            print(translate("\u25c6 LoRAファイル: {0}").format(os.path.basename(all_lora_files[0].name)))
            print(translate("\u25c6 LoRA適用強度: {0}").format(scales[0]))
        elif len(all_lora_files) > 1:
            # 複数ファイル
            print(translate("\u25c6 LoRAファイル (複数):"))
            for i, file in enumerate(all_lora_files):
                print(f"   - {os.path.basename(file.name)} (スケール: {scales[i]})")
        else:
            # LoRAファイルなし
            print(translate("\u25c6 LoRA: 使用しない"))

    # セクションごとのキーフレーム画像の使用状況をログに出力
    valid_sections = []
    if section_settings is not None:
        for i, sec_data in enumerate(section_settings):
            if sec_data and sec_data[1] is not None:  # 画像が設定されている場合
                valid_sections.append(sec_data[0])

    if valid_sections:
        print(translate("\u25c6 使用するキーフレーム画像: セクション{0}").format(', '.join(map(str, valid_sections))))
    else:
        print(translate("◆ キーフレーム画像: デフォルト設定のみ使用"))

    print("=============================\n")

    # バッチ処理の全体停止用フラグ
    eichi_shared.batch_stopped = False

    # 元のシード値を保存（バッチ処理用）
    original_seed = seed

    if use_random_seed:
        seed = random.randint(0, 2**32 - 1)
        # UIのseed欄もランダム値で更新
        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update(value=seed)
        # ランダムシードの場合は最初の値を更新
        original_seed = seed
    else:
        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update()

    eichi_plus_shared.stream = AsyncStream()

    # stream作成後、バッチ処理前もう一度フラグを確認
    if eichi_shared.batch_stopped:
        print(translate("\nバッチ処理が中断されました（バッチ開始前）"))
        yield (
            None,
            gr.update(visible=False),
            translate("バッチ処理が中断されました"),
            '',
            gr.update(interactive=True),
            gr.update(interactive=False, value=translate("End Generation")),
            gr.update()
        )
        return

    # バッチ処理ループの開始
    for batch_index in range(batch_count):
        # 停止フラグが設定されている場合は全バッチ処理を中止
        if eichi_shared.batch_stopped:
            print(translate("\nバッチ処理がユーザーによって中止されました"))
            yield (
                None,
                gr.update(visible=False),
                translate("バッチ処理が中止されました。"),
                '',
                gr.update(interactive=True),
                gr.update(interactive=False, value=translate("End Generation")),
                gr.update()
            )
            break

        # 現在のバッチ番号を表示
        if batch_count > 1:
            batch_info = translate("バッチ処理: {0}/{1}").format(batch_index + 1, batch_count)
            print("\n{batch_info}")
            # UIにもバッチ情報を表示
            yield None, gr.update(visible=False), batch_info, "", gr.update(interactive=False), gr.update(interactive=True), gr.update()

        # バッチインデックスに応じてSEED値を設定
        current_seed = original_seed + batch_index
        if batch_count > 1:
            print(translate("現在のSEED値: {0}").format(current_seed))
        # 現在のバッチ用のシードを設定
        seed = current_seed

        # もう一度停止フラグを確認 - worker処理実行前
        if eichi_shared.batch_stopped:
            print(translate("バッチ処理が中断されました。worker関数の実行をキャンセルします。"))
            # 中断メッセージをUIに表示
            yield (None,
                   gr.update(visible=False),
                   translate("バッチ処理が中断されました（{0}/{1}）").format(batch_index, batch_count),
                   '',
                   gr.update(interactive=True),
                   gr.update(interactive=False, value=translate("End Generation")),
                   gr.update())
            break

        # GPUメモリの設定値をデバッグ出力し、正しい型に変換
        gpu_memory_value = float(gpu_memory_preservation) if gpu_memory_preservation is not None else 6.0
        print(translate('Using GPU memory preservation setting: {0} GB').format(gpu_memory_value))

        # 出力フォルダが空の場合はデフォルト値を使用
        if not output_dir or not output_dir.strip():
            output_dir = "outputs"
        print(translate('Output directory: {0}').format(output_dir))

        # 先に入力データの状態をログ出力（デバッグ用）
        if input_image is not None:
            if isinstance(input_image, str):
                print(translate("[DEBUG] input_image path: {0}, type: {1}").format(input_image, type(input_image)))
            else:
                print(translate("[DEBUG] input_image shape: {0}, type: {1}").format(input_image.shape, type(input_image)))
        if end_frame is not None:
            if isinstance(end_frame, str):
                print(translate("[DEBUG] end_frame path: {0}, type: {1}").format(end_frame, type(end_frame)))
            else:
                print(translate("[DEBUG] end_frame shape: {0}, type: {1}").format(end_frame.shape, type(end_frame)))
        if section_settings is not None:
            print(translate("[DEBUG] section_settings count: {0}").format(len(section_settings)))
            valid_images = sum(1 for s in section_settings if s and s[1] is not None)
            print(translate("[DEBUG] Valid section images: {0}").format(valid_images))

        # バッチ処理の各回で実行
        # worker関数の定義と引数の順序を完全に一致させる
        print(translate("[DEBUG] async_run直前のsave_tensor_data: {0}").format(save_tensor_data))
        async_run(
            worker,
            input_image,
            prompt,
            n_prompt,
            seed,
            total_second_length,
            latent_window_size,
            steps,
            cfg,
            gs,
            rs,
            gpu_memory_value,  # gpu_memory_preservation
            use_teacache,
            mp4_crf,
            all_padding_value,
            end_frame,
            end_frame_strength,
            keep_section_videos,
            lora_files,
            lora_files2,
            lora_scales_text,
            output_dir,
            save_section_frames,
            section_settings,
            use_all_padding,
            use_lora,
            save_tensor_data,  # テンソルデータ保存フラグ - 確実に正しい位置に配置
            tensor_data_input,
            fp8_optimization,
            resolution,
            batch_index
        )

        # 現在のバッチの出力ファイル名
        batch_output_filename = None

        # 現在のバッチの処理結果を取得
        while True:
            flag, data = eichi_plus_shared.stream.output_queue.next()

            if flag == 'file':
                batch_output_filename = data
                # より明確な更新方法を使用し、preview_imageを明示的にクリア
                yield batch_output_filename, gr.update(value=None, visible=False), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update()

            if flag == 'progress':
                preview, desc, html = data
                # バッチ処理中は現在のバッチ情報を追加
                if batch_count > 1:
                    batch_info = translate("バッチ処理: {0}/{1} - ").format(batch_index + 1, batch_count)
                    desc = batch_info + desc
                # preview_imageを明示的に設定
                yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), gr.update()

            if flag == 'end':
                # このバッチの処理が終了
                if batch_index == batch_count - 1 or eichi_shared.batch_stopped:
                    # 最終バッチの場合は処理完了を通知
                    completion_message = ""
                    if eichi_shared.batch_stopped:
                        completion_message = translate("バッチ処理が中止されました（{0}/{1}）").format(batch_index + 1, batch_count)
                    else:
                        completion_message = translate("バッチ処理が完了しました（{0}/{1}）").format(batch_count, batch_count)
                    yield (
                        batch_output_filename,
                        gr.update(value=None, visible=False),
                        completion_message,
                        '',
                        gr.update(interactive=True),
                        gr.update(interactive=False, value=translate("End Generation")),
                        gr.update()
                    )
                else:
                    # 次のバッチに進むメッセージを表示
                    next_batch_message = translate("バッチ処理: {0}/{1} 完了、次のバッチに進みます...").format(batch_index + 1, batch_count)
                    yield (
                        batch_output_filename,
                        gr.update(value=None, visible=False),
                        next_batch_message,
                        '',
                        gr.update(interactive=False),
                        gr.update(interactive=True),
                        gr.update()
                    )
                break

        # 最終的な出力ファイル名を更新
        output_filename = batch_output_filename

        # バッチ処理が停止されている場合はループを抜ける
        if eichi_shared.batch_stopped:
            print(translate("バッチ処理ループを中断します"))
            break


def end_process():
    # 現在のバッチと次のバッチ処理を全て停止するフラグを設定
    eichi_shared.batch_stopped = True
    print(translate("\n停止ボタンが押されました。バッチ処理を停止します..."))
    # 現在実行中のバッチを停止
    eichi_plus_shared.stream.input_queue.push('end')

    # ボタンの名前を一時的に変更することでユーザーに停止処理が進行中であることを表示
    return gr.update(value=translate("停止処理中..."))




def main():
    global mode_radio
    # 既存のQuick Prompts（初期化時にプリセットに変換されるので、互換性のために残す）
    quick_prompts = [
        'A character doing some simple body movements.',
        'A character uses expressive hand gestures and body language.',
        'A character walks leisurely with relaxed movements.',
        'A character performs dynamic movements with energy and flowing motion.',
        'A character moves in unexpected ways, with surprising transitions poses.',
    ]
    quick_prompts = [[x] for x in quick_prompts]


    css = get_app_css()
    block = gr.Blocks(css=css).queue()
    with block:
        gr.HTML('<h1>FramePack<span class="title-suffix">-eichi</span></h1>')

        # デバッグ情報の表示
        # print_keyframe_debug_info()

        # 一番上の行に「生成モード、セクションフレームサイズ、オールパディング、動画長」を配置
        with gr.Row():
            with gr.Column(scale=1):
                mode_radio = gr.Radio(choices=[MODE_TYPE_NORMAL, MODE_TYPE_LOOP], value=MODE_TYPE_NORMAL, label=translate("生成モード"), info=translate("通常：一般的な生成 / ループ：ループ動画用"))
            with gr.Column(scale=1):
                # フレームサイズ切替用のUIコントロール（名前を「セクションフレームサイズ」に変更）
                frame_size_radio = gr.Radio(
                    choices=[translate("1秒 (33フレーム)"), translate("0.5秒 (17フレーム)")],
                    value=translate("1秒 (33フレーム)"),
                    label=translate("セクションフレームサイズ"),
                    info=translate("1秒 = 高品質・通常速度 / 0.5秒 = よりなめらかな動き（実験的機能）")
                )
            with gr.Column(scale=1):
                # オールパディング設定
                use_all_padding = gr.Checkbox(label=translate("オールパディング"), value=False, info=translate("数値が小さいほど直前の絵への影響度が下がり動きが増える"), elem_id="all_padding_checkbox")
                all_padding_value = gr.Slider(label=translate("パディング値"), minimum=0.2, maximum=3, value=1, step=0.1, info=translate("すべてのセクションに適用するパディング値（0.2〜3の整数）"), visible=False)

                # オールパディングのチェックボックス状態に応じてスライダーの表示/非表示を切り替える
                def toggle_all_padding_visibility(use_all_padding):
                    return gr.update(visible=use_all_padding)

                use_all_padding.change(
                    fn=toggle_all_padding_visibility,
                    inputs=[use_all_padding],
                    outputs=[all_padding_value]
                )
            with gr.Column(scale=1):
                # 設定から動的に選択肢を生成
                length_radio = gr.Radio(choices=get_video_modes(), value=translate("1秒"), label=translate("動画長"), info=translate("キーフレーム画像のコピー範囲と動画の長さを設定"))

        with gr.Row():
            with gr.Column():
                # Final Frameの上に説明を追加
                gr.Markdown(translate("**Finalは最後の画像、Imageは最初の画像(最終キーフレーム画像といずれか必須)となります。**"))
                end_frame = gr.Image(sources=['upload', 'clipboard'], type="filepath", label=translate("Final Frame (Optional)"), height=320)

                # End Frame画像のアップロード時のメタデータ抽出機能は一旦コメント化
                # def update_from_end_frame_metadata(image):
                #     """End Frame画像からメタデータを抽出してUIに反映する"""
                #     if image is None:
                #         return [gr.update()] * 2
                #
                #     try:
                #         # NumPy配列からメタデータを抽出
                #         metadata = extract_metadata_from_numpy_array(image)
                #
                #         if not metadata:
                #             print(translate("End Frame画像にメタデータが含まれていません"))
                #             return [gr.update()] * 2
                #
                #         print(translate("End Frame画像からメタデータを抽出しました: {0}").format(metadata))
                #
                #         # プロンプトとSEEDをUIに反映
                #         prompt_update = gr.update()
                #         seed_update = gr.update()
                #
                #         if PROMPT_KEY in metadata and metadata[PROMPT_KEY]:
                #             prompt_update = gr.update(value=metadata[PROMPT_KEY])
                #             print(translate("プロンプトをEnd Frame画像から取得: {0}").format(metadata[PROMPT_KEY]))
                #
                #         if SEED_KEY in metadata and metadata[SEED_KEY]:
                #             # SEED値を整数に変換
                #             try:
                #                 seed_value = int(metadata[SEED_KEY])
                #                 seed_update = gr.update(value=seed_value)
                #                 print(translate("SEED値をEnd Frame画像から取得: {0}").format(seed_value))
                #             except (ValueError, TypeError):
                #                 print(translate("SEED値の変換エラー: {0}").format(metadata[SEED_KEY]))
                #
                #         return [prompt_update, seed_update]
                #     except Exception as e:
                #         print(translate("End Frameメタデータ抽出エラー: {0}").format(e))
                #         return [gr.update()] * 2
                #
                # # End Frame画像アップロード時のメタデータ取得処理を登録
                # end_frame.change(
                #     fn=update_from_end_frame_metadata,
                #     inputs=[end_frame],
                #     outputs=[prompt, seed]
                # )

                # テンソルデータ設定をグループ化して灰色のタイトルバーに変更
                with gr.Group():
                    gr.Markdown(f"### " + translate("テンソルデータ設定"))

                    # テンソルデータ使用有無のチェックボックス
                    use_tensor_data = gr.Checkbox(label=translate("テンソルデータを使用する"), value=False, info=translate("チェックをオンにするとテンソルデータをアップロードできます"))

                    # テンソルデータ設定コンポーネント（初期状態では非表示）
                    with gr.Group(visible=False) as tensor_data_group:
                        tensor_data_input = gr.File(
                            label=translate("テンソルデータアップロード (.safetensors) - 生成動画の後方(末尾)に結合されます"),
                            file_types=[".safetensors"]
                        )

                        gr.Markdown(translate("※ テンソルデータをアップロードすると通常の動画生成後に、その動画の後方（末尾）に結合されます。\n結合した動画は「元のファイル名_combined.mp4」として保存されます。\n※ テンソルデータの保存機能を有効にすると、生成とアップロードのテンソルを結合したデータも保存されます。\n※ テンソルデータの結合は別ツール `python eichi_utils/tensor_combiner.py --ui` でもできます。"))

                    # チェックボックスの状態によってテンソルデータ設定の表示/非表示を切り替える関数
                    def toggle_tensor_data_settings(use_tensor):
                        return gr.update(visible=use_tensor)

                    # チェックボックスの変更イベントに関数を紐づけ
                    use_tensor_data.change(
                        fn=toggle_tensor_data_settings,
                        inputs=[use_tensor_data],
                        outputs=[tensor_data_group]
                    )

                # テンソルデータ設定の下に解像度スライダーとバッチ処理回数を追加
                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=2):
                            resolution = gr.Slider(
                                label=translate("解像度"),
                                minimum=512,
                                maximum=768,
                                value=640,
                                step=128,
                                info=translate("出力動画の基準解像度。現在は512か640か768のいずれかのみ対応（640推奨）")
                            )
                        with gr.Column(scale=1):
                            batch_count = gr.Slider(
                                label=translate("バッチ処理回数"),
                                minimum=1,
                                maximum=100,
                                value=1,
                                step=1,
                                info=translate("同じ設定で連続生成する回数。SEEDは各回で+1されます")
                            )

                # 開始・終了ボタン
                with gr.Row():
                    start_button = gr.Button(value=translate("Start Generation"))
                    end_button = gr.Button(value=translate("End Generation"), interactive=False)

                # セクション入力用のリストを初期化
                section_number_inputs = []
                section_image_inputs = []
                section_prompt_inputs = []  # プロンプト入力欄用のリスト
                section_row_groups = []  # 各セクションのUI行を管理するリスト

                # 設定から最大キーフレーム数を取得
                max_keyframes = get_max_keyframes_count()

                # 現在の動画モードで必要なセクション数を取得する関数
                def get_current_sections_count():
                    mode_value = length_radio.value
                    if mode_value in VIDEO_MODE_SETTINGS:
                        # sections値をそのまま使用 - 注：これは0から始めた場合の最大値となる
                        return VIDEO_MODE_SETTINGS[mode_value]["sections"]
                    return max_keyframes  # デフォルト値

                # 現在の必要セクション数を取得
                initial_sections_count = get_current_sections_count()
                # セクション設定タイトルの定義と動的な更新用の関数
                # 現在のセクション数に応じたMarkdownを返す関数
                def generate_section_title(total_sections):
                    last_section = total_sections - 1
                    return translate('### セクション設定（逆順表示）\n\nセクションは逆時系列で表示されています。Image(始点)は必須でFinal(終点)から遡って画像を設定してください。**最終キーフレームの画像は、Image(始点)より優先されます。総数{0}**').format(total_sections)

                # 動画のモードとフレームサイズに基づいてセクション数を計算し、タイトルを更新する関数
                def update_section_title(frame_size, mode, length):
                    seconds = get_video_seconds(length)
                    latent_window_size = 4.5 if frame_size == translate("0.5秒 (17フレーム)") else 9
                    frame_count = latent_window_size * 4 - 3
                    total_frames = int(seconds * 30)
                    total_sections = int(max(round(total_frames / frame_count), 1))
                    # 表示セクション数の設定
                    # 例: 総セクション数が5の場合、4～0の5個のセクションが表示される
                    display_sections = total_sections
                    return generate_section_title(display_sections)

                # 初期タイトルを計算
                initial_title = update_section_title(translate("1秒 (33フレーム)"), MODE_TYPE_NORMAL, translate("1秒"))

                # 埋め込みプロンプトおよびシードを複写するチェックボックスの定義
                eichi_shared.copy_metadata = gr.Checkbox(
                    label=translate("埋め込みプロンプトおよびシードを複写する"),
                    value=False,
                    info=translate("チェックをオンにすると、画像のメタデータからプロンプトとシードを自動的に取得します")
                )

                with gr.Accordion(translate("セクション設定"), open=False, elem_classes="section-accordion"):
                    # セクション情報zipファイルアップロード処理を追加
                    with gr.Group():
                        gr.Markdown(f"### " + translate("セクション情報一括アップロード"))
                        # チェックボックスで表示/非表示を切り替え
                        show_upload_section = gr.Checkbox(
                            label=translate("一括アップロード機能を表示"),
                            value=False,
                            info=translate("チェックをオンにするとセクション情報の一括アップロード機能を表示します")
                        )
                        # 初期状態では非表示
                        with gr.Group(visible=False) as upload_section_group:
                            upload_zipfile = gr.File(label=translate("セクション情報アップロードファイル"), file_types=[".zip"], interactive=True)

                        # チェックボックスの状態変更時に表示/非表示を切り替える
                        show_upload_section.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[show_upload_section],
                            outputs=[upload_section_group]
                        )

                    with gr.Group(elem_classes="section-container"):
                        section_title = gr.Markdown(initial_title)

                        # セクション番号0の上にコピー機能チェックボックスを追加（ループモード時のみ表示）
                        with gr.Row(visible=(mode_radio.value == MODE_TYPE_LOOP)) as copy_button_row:
                            keyframe_copy_checkbox = gr.Checkbox(label=translate("キーフレーム自動コピー機能を有効にする"), value=True, info=translate("オンにするとキーフレーム間の自動コピーが行われます"))

                        for i in range(max_keyframes):
                            with gr.Row(visible=(i < initial_sections_count), elem_classes="section-row") as row_group:
                                # 左側にセクション番号とプロンプトを配置
                                with gr.Column(scale=1):
                                    section_number = gr.Number(label=translate("セクション番号 {0}").format(i), value=i, precision=0)
                                    section_prompt = gr.Textbox(label=translate("セクションプロンプト {0}").format(i), placeholder=translate("セクション固有のプロンプト（空白の場合は共通プロンプトを使用）"), lines=2)

                                # 右側にキーフレーム画像のみ配置
                                with gr.Column(scale=2):
                                    section_image = gr.Image(label=translate("キーフレーム画像 {0}").format(i), sources="upload", type="filepath", height=200)

                                    # 各キーフレーム画像のアップロード時のメタデータ抽出処理
                                    # クロージャーで現在のセクション番号を捕捉
                                    def create_section_metadata_handler(section_idx, section_prompt_input):
                                        def update_from_section_image_metadata(image_path, copy_enabled=False):
                                            print(translate("\n[DEBUG] セクション{0}の画像メタデータ抽出処理が開始されました").format(section_idx))
                                            print(translate("[DEBUG] メタデータ複写機能: {0}").format(copy_enabled))

                                            # 複写機能が無効の場合は何もしない
                                            if not copy_enabled:
                                                print(translate("[DEBUG] セクション{0}: メタデータ複写機能が無効化されているため、処理をスキップします").format(section_idx))
                                                return gr.update()

                                            if image_path is None:
                                                print(translate("[DEBUG] セクション{0}の画像パスがNoneです").format(section_idx))
                                                return gr.update()

                                            print(translate("[DEBUG] セクション{0}の画像パス: {1}").format(section_idx, image_path))

                                            try:
                                                # ファイルパスから直接メタデータを抽出
                                                print(translate("[DEBUG] セクション{0}からextract_metadata_from_pngを直接呼び出し").format(section_idx))
                                                metadata = extract_metadata_from_png(image_path)

                                                if not metadata:
                                                    print(translate("[DEBUG] セクション{0}の画像からメタデータが抽出されませんでした").format(section_idx))
                                                    return gr.update()

                                                print(translate("[DEBUG] セクション{0}の抽出されたメタデータ: {1}").format(section_idx, metadata))

                                                # セクションプロンプトを取得
                                                if SECTION_PROMPT_KEY in metadata and metadata[SECTION_PROMPT_KEY]:
                                                    section_prompt_value = metadata[SECTION_PROMPT_KEY]
                                                    print(translate("[DEBUG] セクション{0}のプロンプトを画像から取得: {1}").format(section_idx, section_prompt_value))
                                                    print(translate("セクション{0}のプロンプトを画像から取得: {1}").format(section_idx, section_prompt_value))
                                                    return gr.update(value=section_prompt_value)

                                                # 通常のプロンプトがあればそれをセクションプロンプトに設定
                                                elif PROMPT_KEY in metadata and metadata[PROMPT_KEY]:
                                                    prompt_value = metadata[PROMPT_KEY]
                                                    print(translate("[DEBUG] セクション{0}のプロンプトを画像の一般プロンプトから取得: {1}").format(section_idx, prompt_value))
                                                    print(translate("セクション{0}のプロンプトを画像の一般プロンプトから取得: {1}").format(section_idx, prompt_value))
                                                    return gr.update(value=prompt_value)
                                            except Exception as e:
                                                print(translate("[ERROR] セクション{0}のメタデータ抽出エラー: {1}").format(section_idx, e))
                                                traceback.print_exc()
                                                print(translate("セクション{0}のメタデータ抽出エラー: {1}").format(section_idx, e))

                                            return gr.update()
                                        return update_from_section_image_metadata

                                    # キーフレーム画像アップロード時のメタデータ取得処理を登録
                                    section_image.change(
                                        fn=create_section_metadata_handler(i, section_prompt),
                                        inputs=[section_image, eichi_shared.copy_metadata],
                                        outputs=[section_prompt]
                                    )
                                section_number_inputs.append(section_number)
                                section_image_inputs.append(section_image)
                                section_prompt_inputs.append(section_prompt)
                                section_row_groups.append(row_group)  # 行全体をリストに保存

                        # ※ enable_keyframe_copy変数は後で使用するため、ここで定義（モードに応じた初期値設定）
                        enable_keyframe_copy = gr.State(mode_radio.value == MODE_TYPE_LOOP) # ループモードの場合はTrue、通常モードの場合はFalse

                        # キーフレーム自動コピーチェックボックスの変更をenable_keyframe_copyに反映させる関数
                        def update_keyframe_copy_state(value):
                            return value

                        # チェックボックスの変更がenable_keyframe_copyに反映されるようにイベントを設定
                        keyframe_copy_checkbox.change(
                            fn=update_keyframe_copy_state,
                            inputs=[keyframe_copy_checkbox],
                            outputs=[enable_keyframe_copy]
                        )

                        # チェックボックス変更時に赤枠/青枠の表示を切り替える
                        def update_frame_visibility_from_checkbox(value, mode):
                        #   print(translate("チェックボックス変更: 値={0}, モード={1}").format(value, mode))
                            # モードとチェックボックスの両方に基づいて枠表示を決定
                            is_loop = (mode == MODE_TYPE_LOOP)

                            # 通常モードでは常に赤枠/青枠を非表示 (最優先で確認)
                            if not is_loop:
                            #   print(translate("通常モード (チェックボックス値={0}): 赤枠/青枠を強制的に非表示にします").format(value))
                                # 通常モードでは常にelm_classesを空にして赤枠/青枠を非表示に確定する
                                return gr.update(elem_classes=""), gr.update(elem_classes="")

                            # ループモードでチェックボックスがオンの場合のみ枠を表示
                            if value:
                            #   print(translate("ループモード + チェックボックスオン: 赤枠/青枠を表示します"))
                                return gr.update(elem_classes="highlighted-keyframe-red"), gr.update(elem_classes="highlighted-keyframe-blue")
                            else:
                            #   print(translate("ループモード + チェックボックスオフ: 赤枠/青枠を非表示にします"))
                                # ループモードでもチェックがオフなら必ずelem_classesを空にして赤枠/青枠を非表示にする
                                return gr.update(elem_classes=""), gr.update(elem_classes="")

                        keyframe_copy_checkbox.change(
                            fn=update_frame_visibility_from_checkbox,
                            inputs=[keyframe_copy_checkbox, mode_radio],
                            outputs=[section_image_inputs[0], section_image_inputs[1]]
                        )

                        # モード切り替え時にチェックボックスの値と表示状態を制御する
                        def toggle_copy_checkbox_visibility(mode):
                            """モード切り替え時にチェックボックスの表示/非表示を切り替える"""
                            is_loop = (mode == MODE_TYPE_LOOP)
                            # 通常モードの場合はチェックボックスを非表示に設定、コピー機能を必ずFalseにする
                            if not is_loop:
                            #   print(translate("モード切替: {0} -> チェックボックス非表示、コピー機能を無効化").format(mode))
                                return gr.update(visible=False, value=False), gr.update(visible=False), False
                            # ループモードの場合は表示し、デフォルトでオンにする
                        #   print(translate("モード切替: {0} -> チェックボックス表示かつオンに設定").format(mode))
                            return gr.update(visible=True, value=True), gr.update(visible=True), True

                        # モード切り替え時にチェックボックスの表示/非表示と値を制御するイベントを設定
                        mode_radio.change(
                            fn=toggle_copy_checkbox_visibility,
                            inputs=[mode_radio],
                            outputs=[keyframe_copy_checkbox, copy_button_row, enable_keyframe_copy]
                        ) # ループモードに切替時は常にチェックボックスをオンにし、通常モード時は常にオフにする

                        # モード切り替え時に赤枠/青枠の表示を更新
                        def update_frame_visibility_from_mode(mode):
                            # モードに基づいて枠表示を決定
                            is_loop = (mode == MODE_TYPE_LOOP)

                            # 通常モードでは無条件で赤枠/青枠を非表示 (最優先で確定)
                            if not is_loop:
                            #   print(translate("モード切替: 通常モード -> 枠を強制的に非表示"))
                                return gr.update(elem_classes=""), gr.update(elem_classes="")
                            else:
                                # ループモードではチェックボックスが常にオンになるので枠を表示
                            #   print(translate("モード切替: ループモード -> チェックボックスオンなので枠を表示"))
                                return gr.update(elem_classes="highlighted-keyframe-red"), gr.update(elem_classes="highlighted-keyframe-blue")

                        mode_radio.change(
                            fn=update_frame_visibility_from_mode,
                            inputs=[mode_radio],
                            outputs=[section_image_inputs[0], section_image_inputs[1]]
                        )

                input_image = gr.Image(sources=['upload', 'clipboard'], type="filepath", label="Image", height=320)

                # メタデータ抽出関数を定義（後で登録する）
                def update_from_image_metadata(image_path, copy_enabled=False):
                    """Imageアップロード時にメタデータを抽出してUIに反映する
                    copy_enabled: メタデータの複写が有効化されているかどうか
                    """
                    print("\n[DEBUG] update_from_image_metadata関数が実行されました")
                    print(translate("[DEBUG] メタデータ複写機能: {0}").format(copy_enabled))

                    # 複写機能が無効の場合は何もしない
                    if not copy_enabled:
                        print("[DEBUG] メタデータ複写機能が無効化されているため、処理をスキップします")
                        return [gr.update()] * 2

                    if image_path is None:
                        print("[DEBUG] image_pathはNoneです")
                        return [gr.update()] * 2

                    print(translate("[DEBUG] 画像パス: {0}").format(image_path))

                    try:
                        # ファイルパスから直接メタデータを抽出
                        print("[DEBUG] extract_metadata_from_pngをファイルパスから直接呼び出します")
                        metadata = extract_metadata_from_png(image_path)

                        if not metadata:
                            print("[DEBUG] メタデータが抽出されませんでした")
                            print(translate("アップロードされた画像にメタデータが含まれていません"))
                            return [gr.update()] * 2

                        print(translate("[DEBUG] メタデータサイズ: {0}, 内容: {1}").format(len(metadata), metadata))
                        print(translate("画像からメタデータを抽出しました: {0}").format(metadata))

                        # プロンプトとSEEDをUIに反映
                        prompt_update = gr.update()
                        seed_update = gr.update()

                        if PROMPT_KEY in metadata and metadata[PROMPT_KEY]:
                            prompt_update = gr.update(value=metadata[PROMPT_KEY])
                            print(translate("[DEBUG] プロンプトを更新: {0}").format(metadata[PROMPT_KEY]))
                            print(translate("プロンプトを画像から取得: {0}").format(metadata[PROMPT_KEY]))

                        if SEED_KEY in metadata and metadata[SEED_KEY]:
                            # SEED値を整数に変換
                            try:
                                seed_value = int(metadata[SEED_KEY])
                                seed_update = gr.update(value=seed_value)
                                print(translate("[DEBUG] SEED値を更新: {0}").format(seed_value))
                                print(translate("SEED値を画像から取得: {0}").format(seed_value))
                            except (ValueError, TypeError):
                                print(translate("[DEBUG] SEED値の変換エラー: {0}").format(metadata[SEED_KEY]))
                                print(translate("SEED値の変換エラー: {0}").format(metadata[SEED_KEY]))

                        print(translate("[DEBUG] 更新結果: prompt_update={0}, seed_update={1}").format(prompt_update, seed_update))
                        return [prompt_update, seed_update]
                    except Exception as e:
                        print(translate("[ERROR] メタデータ抽出処理中のエラー: {0}").format(e))
                        traceback.print_exc()
                        print(translate("メタデータ抽出エラー: {0}").format(e))
                        return [gr.update()] * 2

                # 注意: イベント登録は変数定義後に行うため、後で実行する
                # メタデータ抽出処理の登録は、promptとseed変数の定義後に移動します

                # LoRA設定グループを追加
                with gr.Group(visible=eichi_shared.has_lora_support) as lora_settings_group:
                    gr.Markdown(f"### " + translate("LoRA設定"))

                    # LoRA使用有無のチェックボックス
                    use_lora = gr.Checkbox(label=translate("LoRAを使用する"), value=False, info=translate("チェックをオンにするとLoRAを使用します（要16GB VRAM以上）"))

                    # LoRA設定コンポーネント（初期状態では非表示）
                    # メインのLoRAファイル
                    lora_files = gr.File(
                        label=translate("LoRAファイル (.safetensors, .pt, .bin)"),
                        file_types=[".safetensors", ".pt", ".bin"],
                        visible=False
                    )
                    # 追加のLoRAファイル
                    lora_files2 = gr.File(
                        label=translate("LoRAファイル2 (.safetensors, .pt, .bin)"),
                        file_types=[".safetensors", ".pt", ".bin"],
                        visible=False
                    )
                    # スケール値の入力フィールド
                    lora_scales_text = gr.Textbox(
                        label=translate("LoRA適用強度 (カンマ区切り)"),
                        value="0.8,0.8",
                        info=translate("各LoRAのスケール値をカンマ区切りで入力 (例: 0.8,0.5)"),
                        visible=False
                    )
                    fp8_optimization = gr.Checkbox(
                        label=translate("FP8最適化"),
                        value=False,
                        info=translate("メモリ使用量を削減し、速度を改善します（PyTorch 2.1以上が必要）"),
                        visible=False
                    )
                    lora_blocks_type = gr.Dropdown(
                        label=translate("LoRAブロック選択"),
                        choices=["all", "single_blocks", "double_blocks", "db0-9", "db10-19", "sb0-9", "sb10-19", "important"],
                        value="all",
                        info=translate("選択するブロックタイプ（all=すべて、その他=メモリ節約）"),
                        visible=False
                    )

                    # チェックボックスの状態によって他のLoRA設定の表示/非表示を切り替える関数
                    def toggle_lora_settings(use_lora):
                        return [
                            gr.update(visible=use_lora),  # lora_files
                            gr.update(visible=use_lora),  # lora_files2
                            gr.update(visible=use_lora),  # lora_scales_text
                            gr.update(visible=use_lora),  # fp8_optimization
                        ]

                    # チェックボックスの変更イベントに関数を紋づけ
                    use_lora.change(fn=toggle_lora_settings,
                            inputs=[use_lora],
                            outputs=[lora_files, lora_files2, lora_scales_text, fp8_optimization])

                    # LoRAサポートが無効の場合のメッセージ
                    if not eichi_shared.has_lora_support:
                        gr.Markdown(translate("LoRAサポートは現在無効です。lora_utilsモジュールが必要です。"))

                prompt = gr.Textbox(label=translate("Prompt"), value=get_default_startup_prompt(), lines=6)

                # プロンプト管理パネルの追加
                with gr.Group(visible=True) as prompt_management:
                    gr.Markdown(f"### " + translate("プロンプト管理"))

                    # 編集画面を常時表示する
                    with gr.Group(visible=True):
                        # 起動時デフォルトの初期表示用に取得
                        default_prompt = ""
                        default_name = ""
                        for preset in load_presets()["presets"]:
                            if preset.get("is_startup_default", False):
                                default_prompt = preset["prompt"]
                                default_name = preset["name"]
                                break

                        with gr.Row():
                            edit_name = gr.Textbox(label=translate("プリセット名"), placeholder=translate("名前を入力..."), value=default_name)

                        edit_prompt = gr.Textbox(label=translate("プロンプト"), lines=5, value=default_prompt)

                        with gr.Row():
                            # 起動時デフォルトをデフォルト選択に設定
                            default_preset = translate("起動時デフォルト")
                            # プリセットデータから全プリセット名を取得
                            presets_data = load_presets()
                            choices = [preset["name"] for preset in presets_data["presets"]]
                            default_presets = [name for name in choices if any(p["name"] == name and p.get("is_default", False) for p in presets_data["presets"])]
                            user_presets = [name for name in choices if name not in default_presets]
                            sorted_choices = [(name, name) for name in sorted(default_presets) + sorted(user_presets)]
                            preset_dropdown = gr.Dropdown(label=translate("プリセット"), choices=sorted_choices, value=default_preset, type="value")

                        with gr.Row():
                            save_btn = gr.Button(value=translate("保存"), variant="primary")
                            apply_preset_btn = gr.Button(value=translate("反映"), variant="primary")
                            clear_btn = gr.Button(value=translate("クリア"))
                            delete_preset_btn = gr.Button(value=translate("削除"))

                    # メッセージ表示用
                    result_message = gr.Markdown("")

                # プリセットの説明文を削除

                # 互換性のためにQuick Listも残しておくが、非表示にする
                with gr.Row(visible=False):
                    example_quick_prompts = gr.Dataset(samples=quick_prompts, label=translate("Quick List"), samples_per_page=1000, components=[prompt])
                    example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

                # 以下の設定ブロックは右カラムに移動しました

                    # セクション設定のリストは既にアコーディオン内で初期化されています
                    # section_number_inputs
                    # section_image_inputs
                    # section_prompt_inputs
                    # section_row_groups

                    # section_settingsは入力欄の値をまとめてリスト化
                    def collect_section_settings(*args):
                        # args: [num1, img1, prompt1, num2, img2, prompt2, ...]
                        return [[args[i], args[i+1], args[i+2]] for i in range(0, len(args), 3)]

                    section_settings = gr.State([[None, None, ""] for _ in range(max_keyframes)])
                    section_inputs = []
                    for i in range(max_keyframes):
                        section_inputs.extend([section_number_inputs[i], section_image_inputs[i], section_prompt_inputs[i]])

                    # section_inputsをまとめてsection_settings Stateに格納
                    def update_section_settings(*args):
                        return collect_section_settings(*args)

                    # section_inputsが変化したらsection_settings Stateを更新
                    for inp in section_inputs:
                        inp.change(fn=update_section_settings, inputs=section_inputs, outputs=section_settings)

                    # フレームサイズ変更時の処理を追加
                    def update_section_calculation(frame_size, mode, length):
                        """フレームサイズ変更時にセクション数を再計算して表示を更新"""
                        # 動画長を取得
                        seconds = get_video_seconds(length)

                        # latent_window_sizeを設定
                        latent_window_size = 4.5 if frame_size == translate("0.5秒 (17フレーム)") else 9
                        frame_count = latent_window_size * 4 - 3

                        # セクション数を計算
                        total_frames = int(seconds * 30)
                        total_sections = int(max(round(total_frames / frame_count), 1))

                        # 計算詳細を表示するHTMLを生成
                        html = f"""<div style='padding: 10px; background-color: #f5f5f5; border-radius: 5px; font-size: 14px;'>
                        {translate('<strong>計算詳細</strong>: モード={0}, フレームサイズ={1}, 総フレーム数={2}, セクションあたり={3}フレーム, 必要セクション数={4}').format(length, frame_size, total_frames, frame_count, total_sections)}
                        <br>
                        {translate('動画モード {0} とフレームサイズ {1} で必要なセクション数: <strong>{2}</strong>').format(length, frame_size, total_sections)}
                        </div>"""

                        # デバッグ用ログ
                        print(translate("計算結果: モード={0}, フレームサイズ={1}, latent_window_size={2}, 総フレーム数={3}, 必要セクション数={4}").format(length, frame_size, latent_window_size, total_frames, total_sections))

                        return html

                    # 初期化時にも計算を実行
                    initial_html = update_section_calculation(frame_size_radio.value, mode_radio.value, length_radio.value)
                    section_calc_display = gr.HTML(value=initial_html, label="")

                    # フレームサイズ変更イベント - HTML表示の更新とセクションタイトルの更新を行う
                    frame_size_radio.change(
                        fn=update_section_calculation,
                        inputs=[frame_size_radio, mode_radio, length_radio],
                        outputs=[section_calc_display]
                    )

                    # フレームサイズ変更時にセクションタイトルも更新
                    frame_size_radio.change(
                        fn=update_section_title,
                        inputs=[frame_size_radio, mode_radio, length_radio],
                        outputs=[section_title]
                    )

                    # セクションの表示/非表示のみを制御する関数
                    def update_section_visibility(mode, length, frame_size=None):
                        """画像は初期化せずにセクションの表示/非表示のみを制御する関数"""
                        # フレームサイズに基づくセクション数計算
                        seconds = get_video_seconds(length)
                        latent_window_size_value = 4.5 if frame_size == translate("0.5秒 (17フレーム)") else 9
                        frame_count = latent_window_size_value * 4 - 3
                        total_frames = int(seconds * 30)
                        total_sections = int(max(round(total_frames / frame_count), 1))

                        # 通常モードの場合は全てのセクションの赤枠青枠を強制的にクリア
                        is_normal_mode = (mode == MODE_TYPE_NORMAL)
                        section_image_updates = []

                        print(translate("セクション視認性更新: モード={mode}, 長さ={length}, 必要セクション数={total_sections}").format(mode=mode, length=length, total_sections=total_sections))

                        for i in range(len(section_image_inputs)):
                            if is_normal_mode:
                                # 通常モードではすべてのセクション画像の赤枠青枠を強制的にクリア
                                # 重要: 通常モードでは無条件で済む結果を返す
                                # print(translate("  セクション{i}: 通常モードなので赤枠/青枠を強制的にクリア").format(i=i))
                                section_image_updates.append(gr.update(elem_classes=""))  # 必ずelem_classesを空に設定
                            else:
                                # ループモードではセクション0と1に赤枠青枠を設定
                                # ループモードではチェックボックスが常にオンになることを利用
                                if i == 0:
                                    # print(translate("  セクション{i}: ループモードのセクション0に赤枠を設定").format(i=i))
                                    section_image_updates.append(gr.update(elem_classes="highlighted-keyframe-red"))
                                elif i == 1:
                                    # print(translate("  セクション{i}: ループモードのセクション1に青枠を設定").format(i=i))
                                    section_image_updates.append(gr.update(elem_classes="highlighted-keyframe-blue"))
                                else:
                                    # print(translate("  セクション{i}: ループモードの他セクションは空枠に設定").format(i=i))
                                    section_image_updates.append(gr.update(elem_classes=""))

                        # 各セクションの表示/非表示のみを更新
                        section_row_updates = []
                        for i in range(len(section_row_groups)):
                            section_row_updates.append(gr.update(visible=(i < total_sections)))

                        # 返値の設定 - input_imageとend_frameは更新せず
                        return [gr.update()] * 2 + section_image_updates + [gr.update(value=seconds)] + section_row_updates

                    # 注意: この関数のイベント登録は、total_second_lengthのUIコンポーネント定義後に行うため、
                    # ここでは関数の定義のみ行い、実際のイベント登録はUIコンポーネント定義後に行います。

                    # 動画長変更イベントでもセクション数計算を更新
                    length_radio.change(
                        fn=update_section_calculation,
                        inputs=[frame_size_radio, mode_radio, length_radio],
                        outputs=[section_calc_display]
                    )

                    # 動画長変更時にセクションタイトルも更新
                    length_radio.change(
                        fn=update_section_title,
                        inputs=[frame_size_radio, mode_radio, length_radio],
                        outputs=[section_title]
                    )

                    # モード変更時にも計算を更新
                    mode_radio.change(
                        fn=update_section_calculation,
                        inputs=[frame_size_radio, mode_radio, length_radio],
                        outputs=[section_calc_display]
                    )

                    # モード変更時にセクションタイトルも更新
                    mode_radio.change(
                        fn=update_section_title,
                        inputs=[frame_size_radio, mode_radio, length_radio],
                        outputs=[section_title]
                    )

                    # モード変更時の処理もtotal_second_lengthコンポーネント定義後に行います

                    # 動画長変更時のセクション表示更新もtotal_second_lengthコンポーネント定義後に行います

                    # 入力画像変更時の処理 - ループモード用に復活
                    # 通常モードでセクションにコピーする処理はコメント化したまま
                    # ループモードのLastにコピーする処理のみ復活

                    # 終端フレームハンドラ関数（FinalからImageへのコピーのみ実装）
                    def loop_mode_final_handler(img, mode, length):
                        """end_frameの変更時、ループモードの場合のみコピーを行う関数"""
                        if img is None:
                            # 画像が指定されていない場合は何もしない
                            return gr.update()

                        # ループモードかどうかで処理を分岐
                        if mode == MODE_TYPE_LOOP:
                            # ループモード: ImageにFinalFrameをコピー
                            return gr.update(value=img)  # input_imageにコピー
                        else:
                            # 通常モード: 何もしない
                            return gr.update()

                    # 終端フレームの変更ハンドラを登録
                    end_frame.change(
                        fn=loop_mode_final_handler,
                        inputs=[end_frame, mode_radio, length_radio],
                        outputs=[input_image]
                    )

                    # 各キーフレーム画像の変更イベントを個別に設定
                    # 一度に複数のコンポーネントを更新する代わりに、個別の更新関数を使用
                    def create_single_keyframe_handler(src_idx, target_idx):
                        def handle_single_keyframe(img, mode, length, enable_copy):
                            # ループモード以外では絶対にコピーを行わない
                            if mode != MODE_TYPE_LOOP:
                                # 通常モードでは絶対にコピーしない
                            #   print(translate("通常モードでのコピー要求を拒否: src={src_idx}, target={target_idx}").format(src_idx=src_idx, target_idx=target_idx))
                                return gr.update()

                            # コピー条件をチェック
                            if img is None or not enable_copy:
                                return gr.update()

                            # 現在のセクション数を動的に計算
                            seconds = get_video_seconds(length)
                            # フレームサイズに応じたlatent_window_sizeの調整（ここではUIの設定によらず計算）
                            frame_size = frame_size_radio.value
                            latent_window_size = 4.5 if frame_size == translate("0.5秒 (17フレーム)") else 9
                            frame_count = latent_window_size * 4 - 3
                            total_frames = int(seconds * 30)
                            total_sections = int(max(round(total_frames / frame_count), 1))

                            # 対象セクションが有効範囲を超えている場合はコピーしない(項目数的に+1)
                            if target_idx >= total_sections:
                            #   print(translate("コピー対象セクション{target_idx}が有効範囲({total_sections}まで)を超えています").format(target_idx=target_idx, total_sections=total_sections))
                                return gr.update()

                            # コピー先のチェック - セクション0は偶数番号に、セクション1は奇数番号にコピー
                            if src_idx == 0 and target_idx % 2 == 0 and target_idx != 0:
                                # 詳細ログ出力
                            #   print(translate("赤枠(0)から偶数セクション{target_idx}へのコピー実行 (動的セクション数:{total_sections})").format(target_idx=target_idx, total_sections=total_sections))
                                return gr.update(value=img)
                            elif src_idx == 1 and target_idx % 2 == 1 and target_idx != 1:
                                # 詳細ログ出力
                            #   print(translate("青枠(1)から奇数セクション{target_idx}へのコピー実行 (動的セクション数:{total_sections})").format(target_idx=target_idx, total_sections=total_sections))
                                return gr.update(value=img)

                            # 条件に合わない場合
                            return gr.update()
                        return handle_single_keyframe

                    # アップロードファイルの内容を各セクション、end_frame、start_frameに反映する関数
                    def upload_zipfile_handler(file):
                        if file is None:
                            # ×で削除した場合、全セクションをクリア
                            gr_outputs = []
                            for i in range(0, max_keyframes):
                                # gradio入力フォームの登録順に追加すること
                                gr_outputs.append(i)
                                gr_outputs.append("")
                                gr_outputs.append(None)
                            # end_frame
                            gr_outputs.append(None)
                            # start_frame
                            gr_outputs.append(None)
                            return gr_outputs
                        else:
                            # 一時ディレクトリで処理
                            # temp_dir配下のフォルダを削除（前回アップロードファイルをクリア）
                            if os.path.exists(eichi_shared.temp_dir):
                                for root, dirs, files in os.walk(eichi_shared.temp_dir, topdown=False):
                                    for name in files:
                                        os.remove(os.path.join(root, name))
                                    for name in dirs:
                                        os.rmdir(os.path.join(root, name))
                                os.rmdir(eichi_shared.temp_dir)
                            # zip展開
                            with zipfile.ZipFile(file.name, "r") as zip_ref:
                                zip_ref.extractall(eichi_shared.temp_dir)
                            # 展開されたファイルをフルパスでリストアップ
                            extracted_files = []
                            for root, dirs, files in os.walk(eichi_shared.temp_dir):
                                if len(files) > 0:
                                    extracted_files.extend([os.path.join(root, f) for f in files])
                                    break
                                elif len(dirs) > 0:
                                    zdir0 = os.path.join(root, dirs[0])
                                    extracted_files.extend([os.path.join(zdir0, f) for f in os.listdir(zdir0)])
                                    break

                            # 展開されたファイルのリストを表示
                            # print("展開されたファイル:")
                            # print("  - " + "\n  - ".join(extracted_files))

                            # プロンプトファイルを取得（1つのみ）
                            prompt_file = [f for f in extracted_files if f.endswith("sections.yml") or f.endswith("sections.yaml")][0]

                            # 画像ファイルを取得
                            image_files = [f for f in extracted_files if f.lower().endswith((".png", ".jpeg", ".jpg", ".webp"))]
                            # セクション用画像のファイルを取得しソートする。ファイル名は3桁の0始まりの数字とする。
                            section_image_files = sorted([f for f in image_files if os.path.basename(f)[:3].isdigit()])
                            # end_frame、start_frame向けの画像ファイルを取得
                            end_frame_image_from_zip = None
                            start_frame_image_from_zip = None
                            end_files = [f for f in image_files if os.path.basename(f).lower().startswith("end")]
                            if len(end_files) > 0:
                                end_frame_image_from_zip = end_files[0]
                            start_files = [f for f in image_files if os.path.basename(f).lower().startswith("start")]
                            if len(start_files) > 0:
                                start_frame_image_from_zip = start_files[0]

                            # プロンプトファイルを読み込んでセクションプロンプトに設定
                            with open(prompt_file, "r", encoding="utf-8") as file:
                                prompt_data = yaml.safe_load(file)

                            # セクション入力情報（zipファイルから取得した情報）
                            section_number_list_from_zip = []
                            section_image_list_from_zip = []
                            section_prompt_list_from_zip = []

                            # yamlファイルのsection_infoからプロンプトを抽出してリスト化
                            for section_num in range(0, max_keyframes):
                                section_number_list_from_zip.append(section_num)
                                section_prompt_list_from_zip.append(next((section["prompt"] for section in prompt_data.get("section_info", []) if section.get("section") == section_num), ""))
                                # image_filesからファイル名の先頭番号を抽出して補間
                                image_file_map = {
                                    int(os.path.basename(img_file)[:3]): img_file for img_file in section_image_files
                                }
                                for section_num in section_number_list_from_zip:
                                    if section_num not in image_file_map:
                                        # セクション番号に対応する画像がない場合は補間
                                        image_file_map[section_num] = None
                                # セクション番号順にソートしてリスト化
                                section_image_list_from_zip = [image_file_map[section_num] for section_num in section_number_list_from_zip]
                            print("sections.yamlファイルに従ってセクションに設定します。")

                            # セクションの入力順にする（セクション番号、セクションプロンプト、キーフレーム画像）
                            # 注意：この方式で画像ファイルを更新すると、gradioのtempフォルダへのアップロードは行われず、指定したファイルパス（temp_dir配下）を直接参照する
                            gr_outputs = []
                            for i in range(0, max_keyframes):
                                gr_outputs.append(section_number_list_from_zip[i])
                                gr_outputs.append(section_prompt_list_from_zip[i])
                                gr_outputs.append(section_image_list_from_zip[i])
                            # end_frameを設定
                            if end_frame_image_from_zip:
                                gr_outputs.append(end_frame_image_from_zip)
                            else:
                                gr_outputs.append(None)
                            # start_frameを設定
                            if start_frame_image_from_zip:
                                gr_outputs.append(start_frame_image_from_zip)
                            else:
                                gr_outputs.append(None)

                            # セクションにzipの内容を設定
                            return gr_outputs

                    # ファイルアップロード時のセクション変更
                    gr_outputs = []
                    for i in range(0, max_keyframes):
                        gr_outputs.append(section_number_inputs[i])
                        gr_outputs.append(section_prompt_inputs[i])
                        gr_outputs.append(section_image_inputs[i])
                    # end_frameを設定
                    gr_outputs.append(end_frame)
                    # start_frameを設定
                    gr_outputs.append(input_image)
                    upload_zipfile.change(fn=upload_zipfile_handler, inputs=[upload_zipfile], outputs=gr_outputs)

                    # 各キーフレームについて、影響を受ける可能性のある後続のキーフレームごとに個別のイベントを設定
                    # ここではイベント登録の定義のみ行い、実際の登録はUIコンポーネント定義後に行う

                    # キーフレーム自動コピーの初期値はStateでデフォルトでTrueに設定済み
                    # enable_keyframe_copyは既にTrueに初期化されているのでここでは特に何もしない

                    # モード切り替え時に赤枠/青枠の表示を切り替える関数
                    # トグル関数は不要になったため削除
                    # 代わりにcheckbox値のみに依存するシンプルな条件分岐を各関数で直接実装

            with gr.Column():
                result_video = gr.Video(
                    label=translate("Finished Frames"),
                    autoplay=True,
                    show_share_button=False,
                    height=512,
                    loop=True,
                    format="mp4",
                    interactive=False,
                )
                progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                progress_bar = gr.HTML('', elem_classes='no-generating-animation')
                preview_image = gr.Image(label=translate("Next Latents"), height=200, visible=False)

                # フレームサイズ切替用のUIコントロールは上部に移動したため削除

                # 計算結果を表示するエリア
                section_calc_display = gr.HTML("", label="")

                use_teacache = gr.Checkbox(label=translate('Use TeaCache'), value=True, info=translate('Faster speed, but often makes hands and fingers slightly worse.'))

                # Use Random Seedの初期値
                use_random_seed_default = True
                seed_default = random.randint(0, 2**32 - 1) if use_random_seed_default else 1

                use_random_seed = gr.Checkbox(label=translate("Use Random Seed"), value=use_random_seed_default)

                n_prompt = gr.Textbox(label=translate("Negative Prompt"), value="", visible=False)  # Not used
                seed = gr.Number(label=translate("Seed"), value=seed_default, precision=0)

                # ここで、メタデータ取得処理の登録を移動する
                # ここでは、promptとseedの両方が定義済み
                input_image.change(
                    fn=update_from_image_metadata,
                    inputs=[input_image, eichi_shared.copy_metadata],
                    outputs=[prompt, seed]
                )

                # チェックボックスの変更時に再読み込みを行う
                def check_metadata_on_checkbox_change(copy_enabled, image_path):
                    if not copy_enabled or image_path is None:
                        return [gr.update()] * 2
                    # チェックボックスオン時に、画像があれば再度メタデータを読み込む
                    return update_from_image_metadata(image_path, copy_enabled)

                # セクション画像のメタデータをチェックボックス変更時に再読み込みする関数
                def update_section_metadata_on_checkbox_change(copy_enabled, *section_images):
                    if not copy_enabled:
                        # チェックボックスがオフの場合は何もしない
                        return [gr.update()] * max_keyframes

                    # 各セクションの画像があれば、それぞれのメタデータを再取得する
                    updates = []
                    for i, section_image in enumerate(section_images):
                        if section_image is not None:
                            # セクションメタデータハンドラを直接利用してメタデータを取得
                            # 前に定義したハンドラを再利用するため、仮引数としてNoneを設定
                            handler = create_section_metadata_handler(i, None)
                            # メタデータを取得
                            update = handler(section_image, copy_enabled)
                            updates.append(update)
                        else:
                            updates.append(gr.update())

                    # 不足分を追加
                    while len(updates) < max_keyframes:
                        updates.append(gr.update())

                    return updates[:max_keyframes]

                eichi_shared.copy_metadata.change(
                    fn=check_metadata_on_checkbox_change,
                    inputs=[eichi_shared.copy_metadata, input_image],
                    outputs=[prompt, seed]
                )

                # セクション画像のメタデータを再読み込みするイベントを追加
                eichi_shared.copy_metadata.change(
                    fn=update_section_metadata_on_checkbox_change,
                    inputs=[eichi_shared.copy_metadata] + section_image_inputs,
                    outputs=section_prompt_inputs
                )

                def set_random_seed(is_checked):
                    if is_checked:
                        return random.randint(0, 2**32 - 1)
                    else:
                        return gr.update()
                use_random_seed.change(fn=set_random_seed, inputs=use_random_seed, outputs=seed)

                total_second_length = gr.Slider(label=translate("Total Video Length (Seconds)"), minimum=1, maximum=120, value=1, step=1)
                latent_window_size = gr.Slider(label=translate("Latent Window Size"), minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                steps = gr.Slider(label=translate("Steps"), minimum=1, maximum=100, value=25, step=1, info=translate('Changing this value is not recommended.'))

                cfg = gr.Slider(label=translate("CFG Scale"), minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                gs = gr.Slider(label=translate("Distilled CFG Scale"), minimum=1.0, maximum=32.0, value=10.0, step=0.01, info=translate('Changing this value is not recommended.'))
                rs = gr.Slider(label=translate("CFG Re-Scale"), minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                available_cuda_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
                default_gpu_memory_preservation_gb = 6 if available_cuda_memory_gb >= 20 else (8 if available_cuda_memory_gb > 16 else 10)
                gpu_memory_preservation = gr.Slider(label=translate("GPU Memory to Preserve (GB) (smaller = more VRAM usage)"), minimum=6, maximum=128, value=default_gpu_memory_preservation_gb, step=0.1, info=translate("空けておくGPUメモリ量を指定。小さい値=より多くのVRAMを使用可能=高速、大きい値=より少ないVRAMを使用=安全"))

                # MP4圧縮設定スライダーを追加
                mp4_crf = gr.Slider(label=translate("MP4 Compression"), minimum=0, maximum=100, value=16, step=1, info=translate("数値が小さいほど高品質になります。0は無圧縮。黒画面が出る場合は16に設定してください。"))

                # セクションごとの動画保存チェックボックスを追加（デフォルトOFF）
                keep_section_videos = gr.Checkbox(label=translate("完了時にセクションごとの動画を残す - チェックがない場合は最終動画のみ保存されます（デフォルトOFF）"), value=False)

                # テンソルデータ保存チェックボックス违加
                save_tensor_data = gr.Checkbox(
                    label=translate("完了時にテンソルデータ(.safetensors)も保存 - このデータを別の動画の後に結合可能"),
                    value=False,
                    info=translate("チェックすると、生成されたテンソルデータを保存します。アップロードされたテンソルがあれば、結合したテンソルデータも保存されます。")
                )

                # セクションごとの静止画保存チェックボックスを追加（デフォルトOFF）
                save_section_frames = gr.Checkbox(label=translate("Save Section Frames"), value=False, info=translate("各セクションの最終フレームを静止画として保存します（デフォルトOFF）"))

                # UIコンポーネント定義後のイベント登録
                # mode_radio.changeの登録 - セクションの表示/非表示と赤枠青枠の表示を同時に更新
                mode_radio.change(
                    fn=update_section_visibility,
                    inputs=[mode_radio, length_radio, frame_size_radio],
                    outputs=[input_image, end_frame] + section_image_inputs + [total_second_length] + section_row_groups
                )

                # frame_size_radio.changeの登録 - セクションの表示/非表示のみを更新
                frame_size_radio.change(
                    fn=update_section_visibility,
                    inputs=[mode_radio, length_radio, frame_size_radio],
                    outputs=[input_image, end_frame] + section_image_inputs + [total_second_length] + section_row_groups
                )

                # length_radio.changeの登録 - セクションの表示/非表示のみを更新
                length_radio.change(
                    fn=update_section_visibility,
                    inputs=[mode_radio, length_radio, frame_size_radio],
                    outputs=[input_image, end_frame] + section_image_inputs + [total_second_length] + section_row_groups
                )

                # mode_radio.changeの登録 - 拡張モード変更ハンドラを使用
                mode_radio.change(
                    fn=lambda mode, length: extended_mode_length_change_handler(mode, length, section_number_inputs, section_row_groups),
                    inputs=[mode_radio, length_radio],
                    outputs=[input_image, end_frame] + section_image_inputs + [total_second_length] + section_row_groups
                )


                # EndFrame影響度調整スライダー
                with gr.Group():
                    gr.Markdown(f"### " + translate("EndFrame影響度調整"))
                    end_frame_strength = gr.Slider(
                        label=translate("EndFrame影響度"),
                        minimum=0.01,
                        maximum=1.00,
                        value=1.00,
                        step=0.01,
                        info=translate("最終フレームが動画全体に与える影響の強さを調整します。値を小さくすると最終フレームの影響が弱まり、最初のフレームに早く移行します。1.00が通常の動作です。")
                    )

                # 出力フォルダ設定
                gr.Markdown(translate("※ 出力先は `webui` 配下に限定されます"))
                with gr.Row(equal_height=True):
                    with gr.Column(scale=4):
                        # フォルダ名だけを入力欄に設定
                        output_dir = gr.Textbox(
                            label=translate("出力フォルダ名"),
                            value=eichi_shared.output_folder_name,  # 設定から読み込んだ値を使用
                            info=translate("動画やキーフレーム画像の保存先フォルダ名"),
                            placeholder="outputs"
                        )
                    with gr.Column(scale=1, min_width=100):
                        open_folder_btn = gr.Button(value=translate("📂 保存および出力フォルダを開く"), size="sm")

                # 実際の出力パスを表示
                with gr.Row(visible=False):
                    path_display = gr.Textbox(
                        label=translate("出力フォルダの完全パス"),
                        value=os.path.join(eichi_shared.base_path, eichi_shared.output_folder_name),
                        interactive=False
                    )

                # フォルダを開くボタンのイベント
                def handle_open_folder_btn(folder_name):
                    """フォルダ名を保存し、そのフォルダを開く"""
                    if not folder_name or not folder_name.strip():
                        folder_name = "outputs"

                    # フォルダパスを取得
                    folder_path = get_output_folder_path(folder_name)

                    # 設定を更新して保存
                    settings = load_settings()
                    old_folder_name = settings.get('output_folder')

                    if old_folder_name != folder_name:
                        settings['output_folder'] = folder_name
                        save_result = save_settings(settings)
                        if save_result:
                            eichi_shared.output_folder_name = folder_name
                            eichi_shared.outputs_folder = folder_path
                        print(translate("出力フォルダ設定を保存しました: {folder_name}").format(folder_name=folder_name))

                    # フォルダを開く
                    open_output_folder(folder_path)

                    # 出力ディレクトリ入力欄とパス表示を更新
                    return gr.update(value=folder_name), gr.update(value=folder_path)

                open_folder_btn.click(fn=handle_open_folder_btn, inputs=[output_dir], outputs=[output_dir, path_display])

                # プロンプト管理パネル（右カラムから左カラムに移動済み）

        # 実行前のバリデーション関数
        def validate_and_process(*args):
            """入力画像または最後のキーフレーム画像のいずれかが有効かどうかを確認し、問題がなければ処理を実行する"""
            input_img = args[0]  # 入力の最初が入力画像
            section_settings = args[24]  # section_settingsはprocess関数の24番目の引数
            resolution_value = args[30] if len(args) > 30 else 640  # resolutionは30番目
            batch_count = args[31] if len(args) > 31 else 1  # batch_countは31番目

            # バッチ回数を有効な範囲に制限
            batch_count = max(1, min(int(batch_count), 100))

            # section_settingsがブール値の場合は空のリストで初期化
            if isinstance(section_settings, bool):
                print(f"[DEBUG] section_settings is bool ({section_settings}), initializing as empty list")
                section_settings = [[None, None, ""] for _ in range(50)]

            # 現在の動画長設定とフレームサイズ設定を渡す
            is_valid, error_message = validate_images(input_img, section_settings, length_radio, frame_size_radio)

            if not is_valid:
                # 画像が無い場合はエラーメッセージを表示して終了
                yield None, gr.update(visible=False), translate("エラー: 画像が選択されていません"), error_message, gr.update(interactive=True), gr.update(interactive=False), gr.update()
                return

            # 画像がある場合は通常の処理を実行
            # 修正したsection_settingsとbatch_countでargsを更新
            new_args = list(args)
            new_args[24] = section_settings  # section_settingsはprocess関数の24番目の引数

            # resolution_valueが整数であることを確認
            try:
                resolution_int = int(float(resolution_value))
                resolution_value = resolution_int
            except (ValueError, TypeError):
                resolution_value = 640
                
            if len(new_args) <= 30:
                # 不足している場合は追加
                if len(new_args) <= 29:
                    # fp8_optimizationがない場合
                    new_args.append(False)
                # resolutionを追加
                new_args.append(resolution_value)
                # batch_countを追加
                new_args.append(batch_count)
            else:
                # 既に存在する場合は更新
                new_args[30] = resolution_value  # resolution
                if len(new_args) > 31:
                    new_args[31] = batch_count  # batch_count
                else:
                    new_args.append(batch_count)  # batch_countを追加

            # process関数のジェネレータを返す
            yield from process(*new_args)

        # 実行ボタンのイベント
        ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_random_seed, mp4_crf, all_padding_value, end_frame, end_frame_strength, frame_size_radio, keep_section_videos, lora_files, lora_files2, lora_scales_text, output_dir, save_section_frames, section_settings, use_all_padding, use_lora, save_tensor_data, tensor_data_input, fp8_optimization, resolution, batch_count]
        start_button.click(fn=validate_and_process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, seed])
        end_button.click(fn=end_process, outputs=[end_button])

        # キーフレーム画像変更時のイベント登録
        # セクション0（赤枚)からの自動コピー処理
        for target_idx in range(1, max_keyframes):
            # 偶数セクションにのみコピー
            if target_idx % 2 == 0:  # 偶数先セクション
                single_handler = create_single_keyframe_handler(0, target_idx)
                section_image_inputs[0].change(
                    fn=single_handler,
                    inputs=[section_image_inputs[0], mode_radio, length_radio, enable_keyframe_copy],
                    outputs=[section_image_inputs[target_idx]]
                )

        # セクション1（青枠)からの自動コピー処理
        for target_idx in range(2, max_keyframes):
            # 奇数セクションにのみコピー
            if target_idx % 2 == 1:  # 奇数先セクション
                single_handler = create_single_keyframe_handler(1, target_idx)
                section_image_inputs[1].change(
                    fn=single_handler,
                    inputs=[section_image_inputs[1], mode_radio, length_radio, enable_keyframe_copy],
                    outputs=[section_image_inputs[target_idx]]
                )

        # 注: create_single_keyframe_handler関数はフレームサイズや動画長に基づいた動的セクション数を計算します
        # UIでフレームサイズや動画長を変更すると、動的に計算されたセクション数に従ってコピー処理が行われます

        # プリセット保存ボタンのイベント
        def save_button_click_handler(name, prompt_text):
            """保存ボタンクリック時のハンドラ関数"""

            # 重複チェックと正規化
            if "A character" in prompt_text and prompt_text.count("A character") > 1:
                sentences = prompt_text.split(".")
                if len(sentences) > 0:
                    prompt_text = sentences[0].strip() + "."
                    # 重複を検出したため正規化

            # プリセット保存
            result_msg = save_preset(name, prompt_text)

            # プリセットデータを取得してドロップダウンを更新
            presets_data = load_presets()
            choices = [preset["name"] for preset in presets_data["presets"]]
            default_presets = [n for n in choices if any(p["name"] == n and p.get("is_default", False) for p in presets_data["presets"])]
            user_presets = [n for n in choices if n not in default_presets]
            sorted_choices = [(n, n) for n in sorted(default_presets) + sorted(user_presets)]

            # メインプロンプトは更新しない（保存のみを行う）
            return result_msg, gr.update(choices=sorted_choices), gr.update()

        # 保存ボタンのクリックイベントを接続
        save_btn.click(
            fn=save_button_click_handler,
            inputs=[edit_name, edit_prompt],
            outputs=[result_message, preset_dropdown, prompt]
        )

        # クリアボタン処理
        def clear_fields():
            return gr.update(value=""), gr.update(value="")

        clear_btn.click(
            fn=clear_fields,
            inputs=[],
            outputs=[edit_name, edit_prompt]
        )

        # プリセット読込処理
        def load_preset_handler(preset_name):
            # プリセット選択時に編集欄のみを更新
            for preset in load_presets()["presets"]:
                if preset["name"] == preset_name:
                    return gr.update(value=preset_name), gr.update(value=preset["prompt"])
            return gr.update(), gr.update()

        # プリセット選択時に編集欄に反映
        def load_preset_handler_wrapper(preset_name):
            # プリセット名がタプルの場合も処理する
            if isinstance(preset_name, tuple) and len(preset_name) == 2:
                preset_name = preset_name[1]  # 値部分を取得
            return load_preset_handler(preset_name)

        preset_dropdown.change(
            fn=load_preset_handler_wrapper,
            inputs=[preset_dropdown],
            outputs=[edit_name, edit_prompt]
        )

        # 反映ボタン処理 - 編集画面の内容をメインプロンプトに反映
        def apply_to_prompt(edit_text):
            """編集画面の内容をメインプロンプトに反映する関数"""
            # 編集画面のプロンプトをメインに適用
            return gr.update(value=edit_text)

        # プリセット削除処理
        def delete_preset_handler(preset_name):
            # プリセット名がタプルの場合も処理する
            if isinstance(preset_name, tuple) and len(preset_name) == 2:
                preset_name = preset_name[1]  # 値部分を取得

            result = delete_preset(preset_name)

            # プリセットデータを取得してドロップダウンを更新
            presets_data = load_presets()
            choices = [preset["name"] for preset in presets_data["presets"]]
            default_presets = [name for name in choices if any(p["name"] == name and p.get("is_default", False) for p in presets_data["presets"])]
            user_presets = [name for name in choices if name not in default_presets]
            sorted_names = sorted(default_presets) + sorted(user_presets)
            updated_choices = [(name, name) for name in sorted_names]

            return result, gr.update(choices=updated_choices)

        apply_preset_btn.click(
            fn=apply_to_prompt,
            inputs=[edit_prompt],
            outputs=[prompt]
        )

        delete_preset_btn.click(
            fn=delete_preset_handler,
            inputs=[preset_dropdown],
            outputs=[result_message, preset_dropdown]
        )

    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument("--server", type=str, default='127.0.0.1')
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--inbrowser", action='store_true')
    parser.add_argument("--lang", type=str, default='ja', help="Language: ja, zh-tw, en")
    args, _ = parser.parse_known_args()

    allowed_paths = [os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './outputs')))]

    # 起動コード
    try:
        block.launch(
            server_name=args.server,
            server_port=args.port,
            share=args.share,
            allowed_paths=allowed_paths,
            inbrowser=args.inbrowser,
        )
    except OSError as e:
        if "Cannot find empty port" in str(e):
            print("\n======================================================")
            print(translate("エラー: FramePack-eichiは既に起動しています。"))
            print(translate("同時に複数のインスタンスを実行することはできません。"))
            print(translate("現在実行中のアプリケーションを先に終了してください。"))
            print("======================================================\n")
            input(translate("続行するには何かキーを押してください..."))
        else:
            # その他のOSErrorの場合は元のエラーを表示
            print(translate("\nエラーが発生しました: {e}").format(e=e))
            input(translate("続行するには何かキーを押してください..."))