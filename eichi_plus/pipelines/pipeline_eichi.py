
import os
import time
import traceback  # デバッグログ出力用
import safetensors.torch as sf
import numpy as np
from PIL import Image
import torch
import einops

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

import eichi_plus.utils
from eichi_plus.utils import flush
import eichi_plus.shared.eichi_plus as eichi_plus_shared

from locales.i18n_extended import translate

from eichi_utils.png_metadata import (
    embed_metadata_to_png, PROMPT_KEY, SEED_KEY, SECTION_PROMPT_KEY, SECTION_NUMBER_KEY
)
from eichi_utils.video_mode_settings import get_video_seconds
from eichi_utils.settings_manager import (
    get_output_folder_path,
    load_settings,
    save_settings,
)
from diffusers_helper.utils import save_bcthw_as_mp4, generate_timestamp, resize_and_center_crop
from diffusers_helper.gradio.progress_bar import make_progress_bar_html

from eichi_plus.pipelines.text_encoder.hunyuan_video import HunyuanVideoTextEncoder
from eichi_plus.pipelines.vae.hunyuan_video import HunyuanVideoVAE
from eichi_plus.pipelines.image_encoder.flux_redux_bfl import FluxReduxBflImageEncoder
from eichi_plus.pipelines.sampler.frame_pack_i2v import FramePackI2VSampler
from eichi_plus.diffusers_helper.memory import (
    cpu, gpu, initialize_storage, get_main_memory_free_gb, 
)

class EichiPipline:
    def __init__(self,
            high_vram: bool = False, storage_dir: str = "swap_store",
            output_folder_name = "outputs",
        ):
        self.high_vram = high_vram
        self.storage_dir = initialize_storage(storage_dir)
        self.output_folder_name = output_folder_name

        self.fp8_path = "models/framepack_transfomer_fp8.safetensors"

        self.text_encoder = HunyuanVideoTextEncoder(high_vram, storage_dir)
        self.vae: HunyuanVideoVAE = HunyuanVideoVAE(high_vram)
        self.image_encoder: FluxReduxBflImageEncoder = FluxReduxBflImageEncoder(high_vram)
        self.sampler: FramePackI2VSampler = FramePackI2VSampler(high_vram, fp8_enabled=True, fp8_path=self.fp8_path, storage_dir=storage_dir)

        self.hv_id = "hunyuanvideo-community/HunyuanVideo"
        self.flux_redux_bfl_id = "lllyasviel/flux_redux_bfl"
        self.fp_id = "lllyasviel/FramePackI2V_HY"
    
    def from_pretrained(self, device=cpu):
        self.text_encoder.from_pretrained(self.hv_id, device)
        self.vae.from_pretrained(self.hv_id, device)
        self.image_encoder.from_pretrained(self.flux_redux_bfl_id, device)
        self.sampler.from_pretrained(self.flux_redux_bfl_id, fp8_enabled=True, fp8_path=self.fp8_path, device=device)

    def load_model_as_complete(self, device=gpu):
        self.text_encoder.load_model_as_complete(device)
        self.vae.load_model_as_complete(device)
        self.image_encoder.load_model_as_complete(device)
        self.sampler.move_model_to_device_with_memory_preservation(device)
    
    def unload_complete_models(self):
        self.text_encoder.unload_complete_models()
        self.vae.unload_complete_models()
        self.image_encoder.unload_complete_models()
        self.sampler.unload_complete_models()
    
    def _prepare_settings(
        self, input_image, 
        section_settings, total_second_length, latent_window_size,
        frame_size_setting,
        use_all_padding=False, all_padding_value=1.0,
        output_dir=None,
        batch_index=None,
    ):
        # 入力画像または表示されている最後のキーフレーム画像のいずれかが存在するか確認
        print(translate("[DEBUG] worker内 input_imageの型: {0}").format(type(input_image)))
        if isinstance(input_image, str):
            print(translate("[DEBUG] input_imageはファイルパスです: {0}").format(input_image))
            has_any_image = (input_image is not None)
        else:
            print(translate("[DEBUG] input_imageはファイルパス以外です"))
            has_any_image = (input_image is not None)
        last_visible_section_image = None
        last_visible_section_num = -1

        if not has_any_image and section_settings is not None:
            # 現在の動画長設定から表示されるセクション数を計算
            total_display_sections = None
            try:
                # 動画長を秒数で取得
                seconds = get_video_seconds(total_second_length)

                # フレームサイズ設定からlatent_window_sizeを計算
                current_latent_window_size = 4.5 if frame_size_setting == "0.5秒 (17フレーム)" else 9
                frame_count = current_latent_window_size * 4 - 3

                # セクション数を計算
                total_frames = int(seconds * 30)
                total_display_sections = int(max(round(total_frames / frame_count), 1))
                print(translate("[DEBUG] worker内の現在の設定によるセクション数: {0}").format(total_display_sections))
            except Exception as e:
                print(translate("[ERROR] worker内のセクション数計算エラー: {0}").format(e))

            # 有効なセクション番号を収集
            valid_sections = []
            for section in section_settings:
                if section and len(section) > 1 and section[0] is not None and section[1] is not None:
                    try:
                        section_num = int(section[0])
                        # 表示セクション数が計算されていれば、それ以下のセクションのみ追加
                        if total_display_sections is None or section_num < total_display_sections:
                            valid_sections.append((section_num, section[1]))
                    except (ValueError, TypeError):
                        continue

            # 有効なセクションがあれば、最大の番号（最後のセクション）を探す
            if valid_sections:
                # 番号でソート
                valid_sections.sort(key=lambda x: x[0])
                # 最後のセクションを取得
                last_visible_section_num, last_visible_section_image = valid_sections[-1]
                print(translate("[DEBUG] worker内の最後のキーフレーム確認: セクション{0} (画像あり)").format(last_visible_section_num))
        
        has_any_image = has_any_image or (last_visible_section_image is not None)
        if not has_any_image:
            raise ValueError("入力画像または表示されている最後のキーフレーム画像のいずれかが必要です")

        # 入力画像がない場合はキーフレーム画像を使用
        if input_image is None and last_visible_section_image is not None:
            print(translate("[INFO] 入力画像が指定されていないため、セクション{0}のキーフレーム画像を使用します").format(last_visible_section_num))
            input_image = last_visible_section_image

        # 出力フォルダの設定
        if output_dir and output_dir.strip():
            # 出力フォルダパスを取得
            outputs_folder = get_output_folder_path(output_dir)
            print(translate("出力フォルダを設定: {0}").format(outputs_folder))

            # フォルダ名が現在の設定と異なる場合は設定ファイルを更新
            if output_dir != self.output_folder_name:
                settings = load_settings()
                settings['output_folder'] = output_dir
                if save_settings(settings):
                    self.output_folder_name = output_dir
                    print(translate("出力フォルダ設定を保存しました: {0}").format(output_dir))
        else:
            # デフォルト設定を使用
            outputs_folder = get_output_folder_path(self.output_folder_name)
            print(translate("デフォルト出力フォルダを使用: {0}").format(outputs_folder))

        # フォルダが存在しない場合は作成
        os.makedirs(outputs_folder, exist_ok=True)

        # 既存の計算方法を保持しつつ、設定からセクション数も取得する
        total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
        total_latent_sections = int(max(round(total_latent_sections), 1))

        # 現在のモードを取得（UIから渡された情報から）
        # セクション数を全セクション数として保存
        total_sections = total_latent_sections

        # 現在のバッチ番号が指定されていれば使用する
        batch_suffix = f"_batch{batch_index+1}" if batch_index is not None else ""
        job_id = generate_timestamp() + batch_suffix

        # セクション处理の詳細ログを出力
        if use_all_padding:
            # オールパディングが有効な場合、すべてのセクションで同じ値を使用
            padding_value = round(all_padding_value, 1)  # 小数点1桁に固定（小数点対応）
            latent_paddings = [padding_value] * total_latent_sections
            print(translate("オールパディングを有効化: すべてのセクションにパディング値 {0} を適用").format(padding_value))
        else:
            # 通常のパディング値計算
            latent_paddings = reversed(range(total_latent_sections))
            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
        
        # 全セクション数を事前に計算して保存（イテレータの消費を防ぐため）
        latent_paddings_list = list(latent_paddings)
        total_sections = len(latent_paddings_list)
        latent_paddings = latent_paddings_list  # リストに変換したものを使用

        print(translate("\u25a0 セクション生成詳細:"))
        print(translate("  - 生成予定セクション: {0}").format(latent_paddings))
        frame_count = latent_window_size * 4 - 3
        print(translate("  - 各セクションのフレーム数: 約{0}フレーム (latent_window_size: {1})").format(frame_count, latent_window_size))
        print(translate("  - 合計セクション数: {0}").format(total_sections))

        # セクション設定の前処理
        def get_section_settings_map(section_settings):
            """
            section_settings: DataFrame形式のリスト [[番号, 画像, プロンプト], ...]
            → {セクション番号: (画像, プロンプト)} のdict
            プロンプトやセクション番号のみの設定も許可する
            """
            result = {}
            if section_settings is not None:
                for row in section_settings:
                    if row and len(row) > 0 and row[0] is not None:
                        # セクション番号を取得
                        sec_num = int(row[0])

                        # セクションプロンプトを取得
                        prm = row[2] if len(row) > 2 and row[2] is not None else ""

                        # 画像を取得（ない場合はNone）
                        img = row[1] if len(row) > 1 and row[1] is not None else None

                        # プロンプトまたは画像のどちらかがあればマップに追加
                        if img is not None or (prm is not None and prm.strip() != ""):
                            result[sec_num] = (img, prm)
            return result

        section_map = get_section_settings_map(section_settings)

        return outputs_folder, section_map, job_id, latent_paddings, total_sections

    def _prepare_uploaded_tensor(self, tensor_data_input=None):
        # テンソルデータのアップロードがあれば読み込み
        uploaded_tensor = None
        if tensor_data_input is not None:
            try:
                tensor_path = tensor_data_input.name
                print(translate("テンソルデータを読み込み: {0}").format(os.path.basename(tensor_path)))
                eichi_plus_shared.stream.output_queue.push(
                    ('progress', (None, '', make_progress_bar_html(0, translate('Loading tensor data ...'))))
                )

                # safetensorsからテンソルを読み込み
                tensor_dict = sf.load_file(tensor_path)

                # テンソルに含まれているキーとシェイプを確認
                print(translate("テンソルデータの内容:"))
                for key, tensor in tensor_dict.items():
                    print(f"  - {key}: shape={tensor.shape}, dtype={tensor.dtype}")

                # history_latentsと呼ばれるキーが存在するか確認
                if "history_latents" in tensor_dict:
                    uploaded_tensor = tensor_dict["history_latents"]
                    print(translate("テンソルデータ読み込み成功: shape={0}, dtype={1}").format(uploaded_tensor.shape, uploaded_tensor.dtype))
                    eichi_plus_shared.stream.output_queue.push((
                        'progress',
                        (
                            None, translate('Tensor data loaded successfully!'), 
                            make_progress_bar_html(10, translate('Tensor data loaded successfully!'))
                        )
                    ))
                else:
                    print(translate("警告: テンソルデータに 'history_latents' キーが見つかりません"))
            except Exception as e:
                print(translate("テンソルデータ読み込みエラー: {0}").format(e))
                traceback.print_exc()
        return uploaded_tensor

    def _preprocess_image(self, input_image, prompt, seed, outputs_folder, job_id, resolution=640):
        input_image_np, input_image_pt, height, width = self.vae.preprocess_image(input_image, resolution=resolution)
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))
        # 入力画像にメタデータを埋め込んで保存
        initial_image_path = os.path.join(outputs_folder, f'{job_id}.png')
        Image.fromarray(input_image_np).save(initial_image_path)

        # メタデータの埋め込み
        # print(translate("\n[DEBUG] 入力画像へのメタデータ埋め込み開始: {0}").format(initial_image_path))
        # print(f"[DEBUG] prompt: {prompt}")
        # print(f"[DEBUG] seed: {seed}")
        metadata = {
            PROMPT_KEY: prompt,
            SEED_KEY: seed
        }
        # print(translate("[DEBUG] 埋め込むメタデータ: {0}").format(metadata))
        embed_metadata_to_png(initial_image_path, metadata)

        return input_image_np, input_image_pt, height, width

    def __call__(
        self, input_image, prompt, n_prompt, seed, steps, cfg, gs, rs, 
        gpu_memory_preservation, use_teacache, 
        section_settings, total_second_length, latent_window_size,
        frame_size_setting,
        mp4_crf=16,
        use_all_padding=False, all_padding_value=1.0,
        end_frame=None, end_frame_strength=1.0, 
        keep_section_videos=False, 
        save_tensor_data=False, tensor_data_input=None,
        output_dir=None, save_section_frames=False, 
        batch_index=None,
        resolution=640,
    ):
        outputs_folder, section_map, job_id, latent_paddings, total_sections = self._prepare_settings(
            input_image, 
            section_settings, total_second_length, latent_window_size,
            frame_size_setting,
            use_all_padding, all_padding_value,
            output_dir,
            batch_index,
        )

        # 処理時間計測の開始
        process_start_time = time.time()

        eichi_plus_shared.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

        try:
            section_numbers_sorted = sorted(section_map.keys()) if section_map else []
            eichi_plus_shared.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("Text encoding ...")))))
            # Text encoding
            self.text_encoder.from_pretrained(self.hv_id, cpu)
            if not self.high_vram:
                self.text_encoder.load_model_as_complete(gpu)
            (
                llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, 
                llama_attention_mask, llama_attention_mask_n,
                section_prompt_embeddings,
            ) = self.text_encoder.encode(prompt, n_prompt, cfg, section_map)
            if not self.high_vram:
                self.text_encoder.unload_complete_models()
                flush()
            if any(v is None for v in (
                llama_vec, llama_vec_n, 
                clip_l_pooler, clip_l_pooler_n, 
                llama_attention_mask, llama_attention_mask_n, 
                section_prompt_embeddings
            )):
                raise "テキストエンコード処理に失敗しました。"

            uploaded_tensor = self._prepare_uploaded_tensor(tensor_data_input)

            eichi_plus_shared.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("Image processing ...")))))
            # Image preprocessing
            input_image_np, input_image_pt, height, width = self._preprocess_image(input_image, prompt, seed, outputs_folder, job_id, resolution)

            eichi_plus_shared.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("VAE encoding ...")))))
            # VAE encoding
            self.vae.from_pretrained(self.hv_id, cpu)
            if not self.high_vram:
                self.vae.load_model_as_complete(gpu)
            input_image_np, start_latent, section_latents, end_frame_latent = self.vae.uploaded_tensor(
                input_image, input_image_np, input_image_pt, uploaded_tensor, outputs_folder, job_id, section_map, resolution, end_frame
            )
            if not self.high_vram:
                self.vae.unload_complete_models()
                flush()
            if any(v is None for v in (
                input_image_np, start_latent
            )):
                raise "VAEエンコード処理に失敗しました。"

            eichi_plus_shared.stream.output_queue.push(
                ('progress', (None, '', make_progress_bar_html(0, translate("CLIP Vision encoding ..."))))
            )
            # CLIP Vision
            self.image_encoder.from_pretrained(self.flux_redux_bfl_id, cpu)
            if not self.high_vram:
                self.image_encoder.load_model_as_complete(gpu)
            image_encoder_last_hidden_state = self.image_encoder.encode(input_image_np)
            if not self.high_vram:
                self.image_encoder.unload_complete_models()
                flush()
            if image_encoder_last_hidden_state is None:
                raise "CLIP Visionエンコード処理に失敗しました。"

            eichi_plus_shared.stream.output_queue.push(
                ('progress', (None, '', make_progress_bar_html(0, translate("Start sampling ..."))))
            )
            # Sampling

            self.sampler.from_pretrained(fp8_enabled=True, fp8_path=self.fp8_path)
            llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, image_encoder_last_hidden_state = self.sampler.to_dtype_transfomer(
                llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, image_encoder_last_hidden_state
            )

            rnd = torch.Generator("cpu").manual_seed(seed)
            # latent_window_sizeが4.5の場合は特別に17フレームとする
            if latent_window_size == 4.5:
                num_frames = 17  # 5 * 4 - 3 = 17
            else:
                num_frames = int(latent_window_size * 4 - 3)

            self.vae.init_history_pixels(height, width)
            total_generated_latent_frames = 0

            for i_section, latent_padding in enumerate(latent_paddings):
                # 先に変数を定義
                is_first_section = i_section == 0

                # オールパディングの場合の特別処理
                if use_all_padding:
                    # 最後のセクションの判定
                    is_last_section = i_section == len(latent_paddings) - 1

                    # 内部処理用に元の値を保存
                    orig_padding_value = latent_padding

                    # 最後のセクションが0より大きい場合は警告と強制変換
                    if is_last_section and float(latent_padding) > 0:
                        print(translate("警告: 最後のセクションのパディング値は内部計算のために0に強制します。"))
                        latent_padding = 0
                    elif isinstance(latent_padding, float):
                        # 浮動小数点の場合はそのまま使用（小数点対応）
                        # 小数点1桁に固定のみ行い、丸めは行わない
                        latent_padding = round(float(latent_padding), 1)

                    # 値が変更された場合にデバッグ情報を出力
                    if float(orig_padding_value) != float(latent_padding):
                        print(
                            translate(
                                "パディング値変換: セクション{0}の値を{1}から{2}に変換しました"
                            ).format(i_section, orig_padding_value, latent_padding)
                        )
                else:
                    # 通常モードの場合
                    is_last_section = latent_padding == 0
                
                use_end_latent = is_last_section and end_frame is not None
                latent_padding_size = int(latent_padding * latent_window_size)

                # 定義後にログ出力
                padding_info = translate(
                    "設定パディング値: {0}"
                ).format(all_padding_value) if use_all_padding else translate("パディング値: {0}").format(latent_padding)
                print(translate("\n■ セクション{0}の処理開始 ({1})").format(i_section, padding_info))
                print(translate("  - 現在の生成フレーム数: {0}フレーム").format(total_generated_latent_frames * 4 - 3))
                print(translate("  - 生成予定フレーム数: {0}フレーム").format(num_frames))
                print(translate("  - 最初のセクション?: {0}").format(is_first_section))
                print(translate("  - 最後のセクション?: {0}").format(is_last_section))

                # set current_latent here
                # セクションごとのlatentを使う場合
                if section_map and section_latents is not None and len(section_latents) > 0:
                    # i_section以上で最小のsection_latentsキーを探す
                    valid_keys = [k for k in section_latents.keys() if k >= i_section]
                    if valid_keys:
                        use_key = min(valid_keys)
                        current_latent = section_latents[use_key]
                        print(translate(
                            "[section_latent] section {0}: use section {1} latent (section_map keys: {2})"
                        ).format(
                            i_section, use_key, list(section_latents.keys())
                        ))
                        print(translate(
                            "[section_latent] current_latent id: {0}, min: {1:.4f}, max: {2:.4f}, mean: {3:.4f}"
                        ).format(
                            id(current_latent), current_latent.min().item(), current_latent.max().item(), current_latent.mean().item()
                        ))
                    else:
                        current_latent = start_latent
                        print(translate(
                            "[section_latent] section {0}: use start_latent (no section_latent >= {1})"
                        ).format(
                            i_section, i_section
                        ))
                        print(translate(
                            "[section_latent] current_latent id: {0}, min: {1:.4f}, max: {2:.4f}, mean: {3:.4f}"
                        ).format(
                            id(current_latent), current_latent.min().item(), current_latent.max().item(), current_latent.mean().item()
                        ))
                else:
                    current_latent = start_latent
                    print(translate(
                        "[section_latent] section {0}: use start_latent (no section_latents)"
                    ).format(i_section))
                    print(translate(
                        "[section_latent] current_latent id: {0}, min: {1:.4f}, max: {2:.4f}, mean: {3:.4f}"
                    ).format(
                        id(current_latent), current_latent.min().item(), current_latent.max().item(), current_latent.mean().item()
                    ))

                if is_first_section and end_frame_latent is not None:
                    # EndFrame影響度設定を適用（デフォルトは1.0=通常の影響）
                    if end_frame_strength != 1.0:
                        # 影響度を適用した潜在表現を生成
                        # 値が小さいほど影響が弱まるように単純な乗算を使用
                        # end_frame_strength=1.0のときは1.0倍（元の値）
                        # end_frame_strength=0.01のときは0.01倍（影響が非常に弱い）
                        modified_end_frame_latent = end_frame_latent * end_frame_strength
                        print(translate("EndFrame影響度を{0}に設定（最終フレームの影響が{1}倍）").format(f"{end_frame_strength:.2f}", f"{end_frame_strength:.2f}"))
                        self.vae.history_latents[:, :, 0:1, :, :] = modified_end_frame_latent
                    else:
                        # 通常の処理（通常の影響）
                        self.vae.history_latents[:, :, 0:1, :, :] = end_frame_latent
                
                # セクション固有のプロンプトがあれば使用する（事前にエンコードしたキャッシュを使用）
                (
                    current_llama_vec, current_clip_l_pooler, current_llama_attention_mask
                ) = self.text_encoder.process_section_prompt(
                    i_section, section_map, llama_vec, clip_l_pooler, llama_attention_mask, section_prompt_embeddings
                )

                print(translate('latent_padding_size = {0}, is_last_section = {1}').format(latent_padding_size, is_last_section))

                # latent_window_sizeが4.5の場合は特別に5を使用
                effective_window_size = 5 if latent_window_size == 4.5 else int(latent_window_size)
                indices = torch.arange(0, sum([1, latent_padding_size, effective_window_size, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, effective_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                clean_latents_pre = current_latent.to(self.vae.history_latents)
                (
                    clean_latents_post, clean_latents_2x, clean_latents_4x
                ) = self.vae.history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

                if not self.high_vram:
                    if not self.sampler.model_loaded:
                        self.sampler.from_pretrained(fp8_enabled=True, fp8_path=self.fp8_path)
                    self.sampler.move_model_to_device_with_memory_preservation(
                        gpu, preserved_gpu_memory=gpu_memory_preservation, preserved_cpu_memory=6.0
                    )
                self.sampler.use_teacache(use_teacache, steps)

                generated_latents = self.sampler.sampling(
                    i_section, total_sections, total_generated_latent_frames, steps,
                    height, width, num_frames, cfg, gs, rs, rnd,
                    current_llama_vec, current_llama_attention_mask, current_clip_l_pooler,
                    llama_vec_n, llama_attention_mask_n, clip_l_pooler_n,
                    image_encoder_last_hidden_state, latent_indices, 
                    clean_latents, clean_latent_indices,
                    clean_latents_2x, clean_latent_2x_indices,
                    clean_latents_4x, clean_latent_4x_indices,
                )
                if generated_latents is None:
                    raise "サンプリング処理に失敗しました。"
                
                if not self.high_vram:
                    self.sampler.unload_complete_models()
                    # self.sampler.offload_model_from_device_for_memory_preservation(gpu, preserved_gpu_memory=8.0, preserved_cpu_memory=6.0)
                    flush()

                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

                total_generated_latent_frames += int(generated_latents.shape[2])

                self.vae.from_pretrained(self.hv_id, cpu)
                if not self.high_vram:
                    self.vae.load_model_as_complete(gpu)
                self.vae.add_history_pixels(generated_latents, total_generated_latent_frames, latent_window_size, is_last_section)
                if not self.high_vram:
                    self.vae.unload_complete_models()
                    flush()

                # 各セクションの最終フレームを静止画として保存（セクション番号付き）
                if save_section_frames and self.vae.history_pixels is not None:
                    try:
                        if i_section == 0 or self.vae.current_pixels is None:
                            # 最初のセクションは history_pixels の最後
                            last_frame = self.vae.history_pixels[0, :, -1, :, :]
                        else:
                            # 2セクション目以降は current_pixels の最後
                            last_frame = self.vae.current_pixels[0, :, -1, :, :]
                        last_frame = einops.rearrange(last_frame, 'c h w -> h w c')
                        last_frame = last_frame.cpu().numpy()
                        last_frame = np.clip((last_frame * 127.5 + 127.5), 0, 255).astype(np.uint8)
                        last_frame = resize_and_center_crop(last_frame, target_width=width, target_height=height)

                        # メタデータを埋め込むための情報を収集
                        print(translate("\n[DEBUG] セクション{0}のメタデータ埋め込み準備").format(i_section))
                        section_metadata = {
                            PROMPT_KEY: prompt,  # メインプロンプト
                            SEED_KEY: seed,
                            SECTION_NUMBER_KEY: i_section
                        }
                        print(translate("[DEBUG] 基本メタデータ: {0}").format(section_metadata))

                        # セクション固有のプロンプトがあれば取得
                        if section_map and i_section in section_map:
                            _, section_prompt = section_map[i_section]
                            if section_prompt and section_prompt.strip():
                                section_metadata[SECTION_PROMPT_KEY] = section_prompt
                                print(translate("[DEBUG] セクションプロンプトを追加: {0}").format(section_prompt))

                        # 画像の保存とメタデータの埋め込み
                        if is_first_section and end_frame is None:
                            frame_path = os.path.join(outputs_folder, f'{job_id}_{i_section}_end.png')
                            print(translate("[DEBUG] セクション画像パス: {0}").format(frame_path))
                            Image.fromarray(last_frame).save(frame_path)
                            print(translate("[DEBUG] メタデータ埋め込み実行: {0}").format(section_metadata))
                            embed_metadata_to_png(frame_path, section_metadata)
                        else:
                            frame_path = os.path.join(outputs_folder, f'{job_id}_{i_section}.png')
                            print(translate("[DEBUG] セクション画像パス: {0}").format(frame_path))
                            Image.fromarray(last_frame).save(frame_path)
                            print(translate("[DEBUG] メタデータ埋め込み実行: {0}").format(section_metadata))
                            embed_metadata_to_png(frame_path, section_metadata)

                        print(translate("\u2713 セクション{0}のフレーム画像をメタデータ付きで保存しました").format(i_section))
                    except Exception as e:
                        print(translate("[WARN] セクション{0}最終フレーム画像保存時にエラー: {1}").format(i_section, e))

                if not self.high_vram:
                    self.unload_complete_models()
                
                output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

                save_bcthw_as_mp4(self.vae.history_pixels, output_filename, fps=30, crf=mp4_crf)

                print(translate(
                    'Decoded. Current latent shape {0}; pixel shape {1}'
                ).format(self.vae.real_history_latents.shape, self.vae.history_pixels.shape))

                print(translate("■ セクション{0}の処理完了").format(i_section))
                print(translate("  - 現在の累計フレーム数: {0}フレーム").format(int(max(0, total_generated_latent_frames * 4 - 3))))
                print(translate("  - レンダリング時間: {0}秒").format(f"{max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f}"))
                print(translate("  - 出力ファイル: {0}").format(output_filename))

                eichi_plus_shared.stream.output_queue.push(('file', output_filename))

                if is_last_section:
                    combined_output_filename = None
                    # 全セクション処理完了後、テンソルデータを後方に結合
                    if uploaded_tensor is not None:
                        try:
                            original_frames = real_history_latents.shape[2]  # 元のフレーム数を記録
                            uploaded_frames = uploaded_tensor.shape[2]  # アップロードされたフレーム数

                            print(translate(
                                "テンソルデータを後方に結合します: アップロードされたフレーム数 = {uploaded_frames}"
                            ).format(uploaded_frames=uploaded_frames))
                            # UI上で進捗状況を更新
                            eichi_plus_shared.stream.output_queue.push((
                                'progress', (None, translate(
                                    "テンソルデータ({uploaded_frames}フレーム)の結合を開始します..."
                                ).format(uploaded_frames=uploaded_frames), make_progress_bar_html(80, translate('テンソルデータ結合準備')))
                            ))

                            # テンソルデータを後方に結合する前に、互換性チェック
                            # デバッグログを追加して詳細を出力
                            print(translate(
                                "[DEBUG] テンソルデータの形状: {0}, 生成データの形状: {1}"
                            ).format(uploaded_tensor.shape, real_history_latents.shape))
                            print(translate(
                                "[DEBUG] テンソルデータの型: {0}, 生成データの型: {1}"
                            ).format(uploaded_tensor.dtype, real_history_latents.dtype))
                            print(translate(
                                "[DEBUG] テンソルデータのデバイス: {0}, 生成データのデバイス: {1}"
                            ).format(uploaded_tensor.device, real_history_latents.device))

                            if (
                                uploaded_tensor.shape[3] != real_history_latents.shape[3] 
                                or 
                                uploaded_tensor.shape[4] != real_history_latents.shape[4]
                            ):
                                print(translate("警告: テンソルサイズが異なります: アップロード={0}, 現在の生成={1}").format(uploaded_tensor.shape, real_history_latents.shape))
                                print(translate("テンソルサイズの不一致のため、前方結合をスキップします"))
                                eichi_plus_shared.stream.output_queue.push((
                                    'progress', (None, translate(
                                        "テンソルサイズの不一致のため、前方結合をスキップしました"
                                    ), make_progress_bar_html(85, translate('互換性エラー')))
                                ))
                            else:
                                # デバイスとデータ型を合わせる
                                processed_tensor = uploaded_tensor.clone()
                                if processed_tensor.device != real_history_latents.device:
                                    processed_tensor = processed_tensor.to(real_history_latents.device)
                                if processed_tensor.dtype != real_history_latents.dtype:
                                    processed_tensor = processed_tensor.to(dtype=real_history_latents.dtype)

                                # 元の動画を品質を保ちつつ保存
                                original_output_filename = os.path.join(outputs_folder, f'{job_id}_original.mp4')
                                save_bcthw_as_mp4(history_pixels, original_output_filename, fps=30, crf=mp4_crf)
                                print(translate(
                                    "元の動画を保存しました: {original_output_filename}"
                                ).format(original_output_filename=original_output_filename))

                                # 元データのコピーを取得
                                combined_history_latents = real_history_latents.clone()
                                combined_history_pixels = history_pixels.clone() if history_pixels is not None else None

                                # 各チャンクの処理前に明示的にメモリ解放
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()
                                    import gc
                                    gc.collect()
                                    print(translate(
                                        "[MEMORY] チャンク処理前のGPUメモリ確保状態: {memory:.2f}GB"
                                    ).format(memory=torch.cuda.memory_allocated()/1024**3))

                                # VAEをGPUに移動
                                if not self.high_vram:
                                    print(translate("[SETUP] VAEをGPUに移動: cuda"))
                                    self.vae.from_pretrained()
                                    self.vae.load_model_as_complete(gpu)

                                # 各チャンクを処理
                                # チャンクサイズを設定(各セクションと同等のサイズにする)
                                chunk_size = min(5, uploaded_frames)  # 最大チャンクサイズを5フレームに設定（メモリ使用量を減らすため）

                                # チャンク数を計算
                                num_chunks = (uploaded_frames + chunk_size - 1) // chunk_size

                                # テンソルデータの詳細を出力
                                print(translate("[DEBUG] テンソルデータの詳細分析:"))
                                print(translate("  - 形状: {0}").format(processed_tensor.shape))
                                print(translate("  - 型: {0}").format(processed_tensor.dtype))
                                print(translate("  - デバイス: {0}").format(processed_tensor.device))
                                print(translate(
                                    "  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}"
                                ).format(processed_tensor.min().item(), processed_tensor.max().item(), processed_tensor.mean().item()))
                                print(translate("  - チャンク数: {0}, チャンクサイズ: {1}").format(num_chunks, chunk_size))

                                tensor_size_mb = (processed_tensor.element_size() * processed_tensor.nelement()) / (1024 * 1024)
                                print(translate("  - テンソルデータ全体サイズ: {0:.2f} MB").format(tensor_size_mb))
                                print(translate("  - フレーム数: {0}フレーム（制限無し）").format(uploaded_frames))
                                # 各チャンクを処理
                                for chunk_idx in range(num_chunks):
                                    chunk_start = chunk_idx * chunk_size
                                    chunk_end = min(chunk_start + chunk_size, uploaded_frames)
                                    chunk_frames = chunk_end - chunk_start

                                    # 進捗状況を更新
                                    chunk_progress = (chunk_idx + 1) / num_chunks * 100
                                    progress_message = translate(
                                        "テンソルデータ結合中: チャンク {0}/{1} (フレーム {2}-{3}/{4})"
                                    ).format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames)
                                    eichi_plus_shared.stream.output_queue.push((
                                        'progress', (
                                            None, 
                                            progress_message, 
                                            make_progress_bar_html(int(80 + chunk_progress * 0.1), translate('テンソルデータ処理中'))
                                        )
                                    ))

                                    # 現在のチャンクを取得
                                    current_chunk = processed_tensor[:, :, chunk_start:chunk_end, :, :]
                                    print(translate(
                                        "チャンク{0}/{1}処理中: フレーム {2}-{3}/{4}"
                                    ).format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames))

                                    # メモリ状態を出力
                                    if torch.cuda.is_available():
                                        print(translate(
                                            "[MEMORY] チャンク{0}処理前のGPUメモリ: {1:.2f}GB/{2:.2f}GB"
                                        ).format(
                                            chunk_idx+1, torch.cuda.memory_allocated()/1024**3, 
                                            torch.cuda.get_device_properties(0).total_memory/1024**3
                                        ))
                                        # メモリキャッシュをクリア
                                        torch.cuda.empty_cache()

                                    try:
                                        # 各チャンク処理前にGPUメモリを解放
                                        if torch.cuda.is_available():
                                            torch.cuda.synchronize()
                                            torch.cuda.empty_cache()
                                            import gc
                                            gc.collect()
                                        # チャンクをデコード
                                        # VAEデコードは時間がかかるため、進行中であることを表示
                                        print(translate("チャンク{0}のVAEデコード開始...").format(chunk_idx+1))
                                        eichi_plus_shared.stream.output_queue.push((
                                            'progress', (
                                                None, 
                                                translate("チャンク{0}/{1}のVAEデコード中...").format(chunk_idx+1, num_chunks), 
                                                make_progress_bar_html(int(80 + chunk_progress * 0.1), translate('デコード処理'))
                                            )
                                        ))

                                        # VAEデコード前にテンソル情報を詳しく出力
                                        print(translate("[DEBUG] チャンク{0}のデコード前情報:").format(chunk_idx+1))
                                        print(translate("  - 形状: {0}").format(current_chunk.shape))
                                        print(translate("  - 型: {0}").format(current_chunk.dtype))
                                        print(translate("  - デバイス: {0}").format(current_chunk.device))
                                        print(translate(
                                            "  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}"
                                        ).format(current_chunk.min().item(), current_chunk.max().item(), current_chunk.mean().item()))

                                        # 明示的にデバイスを合わせる
                                        if current_chunk.device != self.vae.vae.device:
                                            print(translate(
                                                "  - デバイスをVAEと同じに変更: {0} → {1}"
                                            ).format(current_chunk.device, self.vae.vae.device))
                                            current_chunk = current_chunk.to(self.vae.vae.device)

                                        # 型を明示的に合わせる
                                        if current_chunk.dtype != torch.float16:
                                            print(translate("  - データ型をfloat16に変更: {0} → torch.float16").format(current_chunk.dtype))
                                            current_chunk = current_chunk.to(dtype=torch.float16)

                                        # VAEデコード処理

                                        chunk_pixels = self.vae.decode(current_chunk).cpu()
                                        if not self.high_vram:
                                            self.vae.unload_complete_models()
                                            flush()
                                        print(translate("チャンク{0}のVAEデコード完了 (フレーム数: {1})").format(chunk_idx+1, chunk_frames))

                                        # デコード後のピクセルデータ情報を出力
                                        print(translate("[DEBUG] チャンク{0}のデコード結果:").format(chunk_idx+1))
                                        print(translate("  - 形状: {0}").format(chunk_pixels.shape))
                                        print(translate("  - 型: {0}").format(chunk_pixels.dtype))
                                        print(translate("  - デバイス: {0}").format(chunk_pixels.device))
                                        print(translate("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(chunk_pixels.min().item(), chunk_pixels.max().item(), chunk_pixels.mean().item()))

                                        # メモリ使用量を出力
                                        if torch.cuda.is_available():
                                            print(translate(
                                                "[MEMORY] チャンク{0}デコード後のGPUメモリ: {1:.2f}GB"
                                            ).format(chunk_idx+1, torch.cuda.memory_allocated()/1024**3))

                                        # 結合する
                                        if combined_history_pixels is None:
                                            # 初回のチャンクの場合はそのまま設定
                                            combined_history_pixels = chunk_pixels
                                        else:
                                            # 2回目以降は結合
                                            print(translate("[DEBUG] 結合前の情報:"))
                                            print(translate(
                                                "  - 既存: {0}, 型: {1}, デバイス: {2}"
                                            ).format(
                                                combined_history_pixels.shape, 
                                                combined_history_pixels.dtype, 
                                                combined_history_pixels.device
                                            ))
                                            print(translate(
                                                "  - 新規: {0}, 型: {1}, デバイス: {2}"
                                            ).format(chunk_pixels.shape, chunk_pixels.dtype, chunk_pixels.device))

                                            # 既存データと新規データで型とデバイスを揃える
                                            if combined_history_pixels.dtype != chunk_pixels.dtype:
                                                print(translate(
                                                    "  - データ型の不一致を修正: {0} → {1}"
                                                ).format(combined_history_pixels.dtype, chunk_pixels.dtype))
                                                combined_history_pixels = combined_history_pixels.to(dtype=chunk_pixels.dtype)

                                            # 両方とも必ずCPUに移動してから結合
                                            if combined_history_pixels.device != torch.device('cpu'):
                                                combined_history_pixels = combined_history_pixels.cpu()
                                            if chunk_pixels.device != torch.device('cpu'):
                                                chunk_pixels = chunk_pixels.cpu()

                                            # 結合処理
                                            combined_history_pixels = torch.cat([combined_history_pixels, chunk_pixels], dim=2)

                                        # 結合後のフレーム数を確認
                                        current_total_frames = combined_history_pixels.shape[2]
                                        print(translate(
                                            "チャンク{0}の結合完了: 現在の組み込みフレーム数 = {1}"
                                        ).format(chunk_idx+1, current_total_frames))

                                        # 中間結果の保存（チャンクごとに保存すると効率が悪いので、最終チャンクのみ保存）
                                        if chunk_idx == num_chunks - 1 or (chunk_idx > 0 and (chunk_idx + 1) % 5 == 0):
                                            # 5チャンクごと、または最後のチャンクで保存
                                            interim_output_filename = os.path.join(
                                                outputs_folder, f'{job_id}_combined_interim_{chunk_idx+1}.mp4'
                                            )
                                            print(translate(
                                                "中間結果を保存中: チャンク{0}/{1}"
                                            ).format(chunk_idx+1, num_chunks))
                                            eichi_plus_shared.stream.output_queue.push((
                                                'progress', (
                                                    None, 
                                                    translate("中間結果のMP4変換中... (チャンク{0}/{1})").format(chunk_idx+1, num_chunks), 
                                                    make_progress_bar_html(int(85 + chunk_progress * 0.1), translate('MP4保存中'))
                                                )
                                            ))

                                            # MP4として保存
                                            save_bcthw_as_mp4(combined_history_pixels, interim_output_filename, fps=30, crf=mp4_crf)
                                            print(translate("中間結果を保存しました: {0}").format(interim_output_filename))

                                            # 結合した動画をUIに反映するため、出力フラグを立てる
                                            eichi_plus_shared.stream.output_queue.push(('file', interim_output_filename))
                                    except Exception as e:
                                        print(translate("チャンク{0}の処理中にエラーが発生しました: {1}").format(chunk_idx+1, e))
                                        traceback.print_exc()

                                        # エラー情報の詳細な出力
                                        print(translate("[ERROR] 詳細エラー情報:"))
                                        print(translate(
                                            "  - チャンク情報: {0}/{1}, フレーム {2}-{3}/{4}"
                                        ).format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames))
                                        if 'current_chunk' in locals():
                                            print(translate(
                                                "  - current_chunk: shape={0}, dtype={1}, device={2}"
                                            ).format(current_chunk.shape, current_chunk.dtype, current_chunk.device))
                                        if 'vae' in globals():
                                            print(translate(
                                                "  - VAE情報: device={0}, dtype={1}"
                                            ).format(self.vae.vae.device, next(self.vae.vae.parameters()).dtype))
                                        
                                        if not self.high_vram:
                                            self.vae.unload_complete_models()
                                            flush()

                                        # GPUメモリ情報
                                        if torch.cuda.is_available():
                                            print(translate(
                                                "  - GPU使用量: {0:.2f}GB/{1:.2f}GB"
                                            ).format(
                                                torch.cuda.memory_allocated()/1024**3,
                                                torch.cuda.get_device_properties(0).total_memory/1024**3
                                            ))

                                        eichi_plus_shared.stream.output_queue.push((
                                            'progress', (
                                                None, translate(
                                                    "エラー: チャンク{0}の処理に失敗しました - {1}"
                                                ).format(chunk_idx+1, str(e)), 
                                                make_progress_bar_html(90, translate('エラー'))
                                            )
                                        ))
                                        break

                                # 処理完了後に明示的にメモリ解放
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                    torch.cuda.empty_cache()
                                    import gc
                                    gc.collect()
                                    print(translate(
                                        "[MEMORY] チャンク処理後のGPUメモリ確保状態: {0:.2f}GB"
                                    ).format(torch.cuda.memory_allocated()/1024**3))

                                # 全チャンクの処理が完了したら、最終的な結合動画を保存
                                if combined_history_pixels is not None:
                                    # 結合された最終結果の情報を出力
                                    print(translate("[DEBUG] 最終結合結果:"))
                                    print(translate("  - 形状: {0}").format(combined_history_pixels.shape))
                                    print(translate("  - 型: {0}").format(combined_history_pixels.dtype))
                                    print(translate("  - デバイス: {0}").format(combined_history_pixels.device))
                                    # 最終結果の保存
                                    print(translate("最終結果を保存中: 全{0}チャンク完了").format(num_chunks))
                                    eichi_plus_shared.stream.output_queue.push((
                                        'progress', (
                                            None, translate("結合した動画をMP4に変換中..."), 
                                            make_progress_bar_html(95, translate('最終MP4変換処理'))
                                        )
                                    ))

                                    # 最終的な結合ファイル名
                                    combined_output_filename = os.path.join(outputs_folder, f'{job_id}_combined.mp4')

                                    # MP4として保存
                                    save_bcthw_as_mp4(combined_history_pixels, combined_output_filename, fps=30, crf=mp4_crf)
                                    print(translate("最終結果を保存しました: {0}").format(combined_output_filename))
                                    print(translate("結合動画の保存場所: {0}").format(os.path.abspath(combined_output_filename)))

                                    # 中間ファイルの削除処理
                                    print(translate("中間ファイルの削除を開始します..."))
                                    deleted_files = []
                                    try:
                                        # 現在のジョブIDに関連する中間ファイルを正規表現でフィルタリング
                                        import re
                                        interim_pattern = re.compile(f'{job_id}_combined_interim_\d+\.mp4')

                                        for filename in os.listdir(outputs_folder):
                                            if interim_pattern.match(filename):
                                                interim_path = os.path.join(outputs_folder, filename)
                                                try:
                                                    os.remove(interim_path)
                                                    deleted_files.append(filename)
                                                    print(translate("  - 中間ファイルを削除しました: {0}").format(filename))
                                                except Exception as e:
                                                    print(translate("  - ファイル削除エラー ({0}): {1}").format(filename, e))

                                        if deleted_files:
                                            print(translate("合計 {0} 個の中間ファイルを削除しました").format(len(deleted_files)))
                                            # 削除ファイル名をユーザーに表示
                                            files_str = ', '.join(deleted_files)
                                            eichi_plus_shared.stream.output_queue.push((
                                                'progress', (
                                                    None, translate("中間ファイルを削除しました: {0}").format(files_str), 
                                                    make_progress_bar_html(97, translate('クリーンアップ完了'))
                                                )
                                            ))
                                        else:
                                            print(translate("削除対象の中間ファイルは見つかりませんでした"))
                                    except Exception as e:
                                        print(translate("中間ファイル削除中にエラーが発生しました: {0}").format(e))
                                        traceback.print_exc()

                                    # 結合した動画をUIに反映するため、出力フラグを立てる
                                    eichi_plus_shared.stream.output_queue.push(('file', combined_output_filename))

                                    # 結合後の全フレーム数を計算して表示
                                    combined_frames = combined_history_pixels.shape[2]
                                    combined_size_mb = (
                                        combined_history_pixels.element_size() * combined_history_pixels.nelement()
                                    ) / (1024 * 1024)

                                    print(translate("結合完了情報: テンソルデータ({0}フレーム) + 新規動画({1}フレーム) = 合計{2}フレーム").format(uploaded_frames, original_frames, combined_frames))
                                    print(translate("結合動画の再生時間: {0:.2f}秒").format(combined_frames / 30))
                                    print(translate("データサイズ: {0:.2f} MB（制限無し）").format(combined_size_mb))

                                    # UI上で完了メッセージを表示
                                    eichi_plus_shared.stream.output_queue.push((
                                        'progress', (
                                            None, 
                                            translate(
                                                "テンソルデータ({0}フレーム)と動画({1}フレーム)の結合が完了しました。\n"
                                                "合計フレーム数: {2}フレーム ({3:.2f}秒) - サイズ制限なし"
                                            ).format(uploaded_frames, original_frames, combined_frames, combined_frames / 30), 
                                            make_progress_bar_html(100, translate('結合完了'))
                                        )
                                    ))
                                else:
                                    print(translate("テンソルデータの結合に失敗しました。"))
                                    eichi_plus_shared.stream.output_queue.push((
                                        'progress', (
                                            None, translate("テンソルデータの結合に失敗しました。"), 
                                            make_progress_bar_html(100, translate('エラー'))
                                        )
                                    ))

                                # 正しく結合された動画はすでに生成済みなので、ここでの処理は不要

                                # この部分の処理はすでに上記のチャンク処理で完了しているため不要

                                # real_history_latentsとhistory_pixelsを結合済みのものに更新
                                real_history_latents = combined_history_latents
                                history_pixels = combined_history_pixels

                                # 結合した動画をUIに反映するため、出力フラグを立てる
                                eichi_plus_shared.stream.output_queue.push(('file', combined_output_filename))

                                # 出力ファイル名を更新
                                output_filename = combined_output_filename

                                # 結合後の全フレーム数を計算して表示
                                combined_frames = combined_history_pixels.shape[2]
                                combined_size_mb = (combined_history_pixels.element_size() * combined_history_pixels.nelement()) / (1024 * 1024)
                                print(translate(
                                    "結合完了情報: テンソルデータ({0}フレーム) + 新規動画({1}フレーム) = 合計{2}フレーム"
                                ).format(uploaded_frames, original_frames, combined_frames))
                                print(translate("結合動画の再生時間: {0:.2f}秒").format(combined_frames / 30))
                                print(translate("データサイズ: {0:.2f} MB（制限無し）").format(combined_size_mb))

                                # UI上で完了メッセージを表示
                                eichi_plus_shared.stream.output_queue.push((
                                    'progress', (
                                        None, translate(
                                            "テンソルデータ({0}フレーム)と動画({1}フレーム)の結合が完了しました。\n"
                                            "合計フレーム数: {2}フレーム ({3:.2f}秒)"
                                        ).format(uploaded_frames, original_frames, combined_frames, combined_frames / 30), 
                                        make_progress_bar_html(100, translate('結合完了'))
                                    )
                                ))
                        except Exception as e:
                            print(translate("テンソルデータ結合中にエラーが発生しました: {0}").format(e))
                            traceback.print_exc()
                            eichi_plus_shared.stream.output_queue.push((
                                'progress', (
                                    None, translate("エラー: テンソルデータ結合に失敗しました - {0}").format(str(e)), 
                                    make_progress_bar_html(100, translate('エラー'))
                                )
                            ))

                    # 処理終了時に通知
                    if HAS_WINSOUND:
                        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                    else:
                        print(translate("\n✓ 処理が完了しました！"))  # Linuxでの代替通知

                    # メモリ解放を明示的に実行
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
                        print(translate(
                            "[MEMORY] 処理完了後のメモリクリア: {memory:.2f}GB/{total_memory:.2f}GB"
                        ).format(
                            memory=torch.cuda.memory_allocated()/1024**3, 
                            total_memory=torch.cuda.get_device_properties(0).total_memory/1024**3
                        ))

                    # テンソルデータの保存処理
                    print(translate("[DEBUG] テンソルデータ保存フラグの値: {0}").format(save_tensor_data))
                    if save_tensor_data:
                        try:
                            # 結果のテンソルを保存するファイルパス
                            tensor_file_path = os.path.join(outputs_folder, f'{job_id}.safetensors')

                            # 保存するデータを準備
                            print(translate("=== テンソルデータ保存処理開始 ==="))
                            print(translate("保存対象フレーム数: {frames}").format(frames=real_history_latents.shape[2]))

                            # サイズ制限を完全に撤廃し、全フレームを保存
                            tensor_to_save = real_history_latents.clone().cpu()

                            # テンソルデータの保存サイズの概算
                            tensor_size_mb = (tensor_to_save.element_size() * tensor_to_save.nelement()) / (1024 * 1024)

                            print(translate("テンソルデータを保存中... shape: {shape}, フレーム数: {frames}, サイズ: {size:.2f} MB").format(shape=tensor_to_save.shape, frames=tensor_to_save.shape[2], size=tensor_size_mb))
                            eichi_plus_shared.stream.output_queue.push((
                                'progress', (
                                    None, translate('テンソルデータを保存中... ({frames}フレーム)').format(frames=tensor_to_save.shape[2]), 
                                    make_progress_bar_html(95, translate('テンソルデータの保存'))
                                )
                            ))

                            # メタデータの準備（フレーム数も含める）
                            metadata = torch.tensor([height, width, tensor_to_save.shape[2]], dtype=torch.int32)

                            # safetensors形式で保存
                            tensor_dict = {
                                "history_latents": tensor_to_save,
                                "metadata": metadata
                            }
                            sf.save_file(tensor_dict, tensor_file_path)

                            print(translate("テンソルデータを保存しました: {path}").format(path=tensor_file_path))
                            print(translate("保存済みテンソルデータ情報: {frames}フレーム, {size:.2f} MB").format(frames=tensor_to_save.shape[2], size=tensor_size_mb))
                            print(translate("=== テンソルデータ保存処理完了 ==="))
                            eichi_plus_shared.stream.output_queue.push((
                                'progress', (
                                    None, translate(
                                        "テンソルデータが保存されました: {path} ({frames}フレーム, {size:.2f} MB)"
                                    ).format(path=os.path.basename(tensor_file_path), frames=tensor_to_save.shape[2], size=tensor_size_mb), 
                                    make_progress_bar_html(100, translate('処理完了'))
                                )
                            ))

                            # アップロードされたテンソルデータがあれば、それも結合したものを保存する
                            if tensor_data_input is not None and uploaded_tensor is not None:
                                try:
                                    # アップロードされたテンソルデータのファイル名を取得
                                    uploaded_tensor_filename = os.path.basename(tensor_data_input.name)
                                    tensor_combined_path = os.path.join(outputs_folder, f'{job_id}_combined_tensors.safetensors')

                                    print(translate("=== テンソルデータ結合処理開始 ==="))
                                    print(translate("生成テンソルと入力テンソルを結合して保存します"))
                                    print(translate("生成テンソル: {frames}フレーム").format(frames=tensor_to_save.shape[2]))
                                    print(translate("入力テンソル: {frames}フレーム").format(frames=uploaded_tensor.shape[2]))

                                    # データ型とデバイスを統一
                                    if uploaded_tensor.dtype != tensor_to_save.dtype:
                                        uploaded_tensor = uploaded_tensor.to(dtype=tensor_to_save.dtype)
                                    if uploaded_tensor.device != tensor_to_save.device:
                                        uploaded_tensor = uploaded_tensor.to(device=tensor_to_save.device)

                                    # サイズチェック
                                    if uploaded_tensor.shape[3] != tensor_to_save.shape[3] or uploaded_tensor.shape[4] != tensor_to_save.shape[4]:
                                        print(translate("警告: テンソルサイズが一致しないため結合できません: {uploaded_shape} vs {tensor_shape}").format(uploaded_shape=uploaded_tensor.shape, tensor_shape=tensor_to_save.shape))
                                    else:
                                        # 結合（生成テンソルの後にアップロードされたテンソルを追加）
                                        combined_tensor = torch.cat([tensor_to_save, uploaded_tensor], dim=2)
                                        combined_frames = combined_tensor.shape[2]
                                        combined_size_mb = (combined_tensor.element_size() * combined_tensor.nelement()) / (1024 * 1024)

                                        # メタデータ更新
                                        combined_metadata = torch.tensor([height, width, combined_frames], dtype=torch.int32)

                                        # 結合したテンソルを保存
                                        combined_tensor_dict = {
                                            "history_latents": combined_tensor,
                                            "metadata": combined_metadata
                                        }
                                        sf.save_file(combined_tensor_dict, tensor_combined_path)

                                        print(translate("結合テンソルを保存しました: {path}").format(path=tensor_combined_path))
                                        print(translate(
                                            "結合テンソル情報: 合計{0}フレーム ({1}+{2}), {3:.2f} MB"
                                        ).format(frames, tensor_to_save.shape[2], uploaded_tensor.shape[2], size))
                                        print(translate("=== テンソルデータ結合処理完了 ==="))
                                        eichi_plus_shared.stream.output_queue.push((
                                            'progress', (
                                                None, translate(
                                                    "テンソルデータ結合が保存されました: 合計{frames}フレーム"
                                                ).format(frames=combined_frames), 
                                                make_progress_bar_html(100, translate('結合テンソル保存完了'))
                                            )
                                        ))
                                except Exception as e:
                                    print(translate("テンソルデータ結合保存エラー: {0}").format(e))
                                    traceback.print_exc()
                        except Exception as e:
                            print(translate("テンソルデータ保存エラー: {0}").format(e))
                            traceback.print_exc()
                            eichi_plus_shared.stream.output_queue.push((
                                'progress', (
                                    None, translate("テンソルデータの保存中にエラーが発生しました。"), 
                                    make_progress_bar_html(100, translate('処理完了'))
                                )
                            ))

                    # 全体の処理時間を計算
                    process_end_time = time.time()
                    total_process_time = process_end_time - process_start_time
                    hours, remainder = divmod(total_process_time, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_str = ""
                    if hours > 0:
                        time_str = translate("{0}時間 {1}分 {2}秒").format(int(hours), int(minutes), f"{seconds:.1f}")
                    elif minutes > 0:
                        time_str = translate("{0}分 {1}秒").format(int(minutes), f"{seconds:.1f}")
                    else:
                        time_str = translate("{0:.1f}秒").format(seconds)
                    print(translate("\n全体の処理時間: {0}").format(time_str))

                    # 完了メッセージの設定（結合有無によって変更）
                    if combined_output_filename is not None:
                        # テンソル結合が成功した場合のメッセージ
                        combined_filename_only = os.path.basename(combined_output_filename)
                        completion_message = translate(
                            "すべてのセクション({sections}/{total_sections})が完了しました。テンソルデータとの後方結合も完了しました。"
                            "結合ファイル名: {filename}\n全体の処理時間: {time}"
                        ).format(sections=sections, total_sections=total_sections, filename=combined_filename_only, time=time_str)
                        # 最終的な出力ファイルを結合したものに変更
                        output_filename = combined_output_filename
                    else:
                        # 通常の完了メッセージ
                        completion_message = translate(
                            "すべてのセクション({sections}/{total_sections})が完了しました。全体の処理時間: {time}"
                            ).format(sections=total_sections, total_sections=total_sections, time=time_str)

                    eichi_plus_shared.stream.output_queue.push((
                        'progress', (None, completion_message, make_progress_bar_html(100, translate('処理完了')))
                    ))

                    # 中間ファイルの削除処理
                    if not keep_section_videos:
                        # 最終動画のフルパス
                        final_video_path = output_filename
                        final_video_name = os.path.basename(final_video_path)
                        # job_id部分を取得（タイムスタンプ部分）
                        job_id_part = job_id

                        # ディレクトリ内のすべてのファイルを取得
                        files = os.listdir(outputs_folder)
                        deleted_count = 0

                        for file in files:
                            # 同じjob_idを持つMP4ファイルかチェック
                            # 結合ファイル('combined'を含む)は消さないように保護
                            if file.startswith(job_id_part) and file.endswith('.mp4') \
                            and file != final_video_name \
                            and 'combined' not in file:  # combinedファイルは保護
                                file_path = os.path.join(outputs_folder, file)
                                try:
                                    os.remove(file_path)
                                    deleted_count += 1
                                    print(translate("[削除] 中間ファイル: {0}").format(file))
                                except Exception as e:
                                    print(translate("[エラー] ファイル削除時のエラー {0}: {1}").format(file, e))

                        if deleted_count > 0:
                            print(translate(
                                "[済] {0}個の中間ファイルを削除しました。最終ファイルは保存されています: {1}"
                            ).format(deleted_count, final_video_name))
                            final_message = translate("中間ファイルを削除しました。最終動画と結合動画は保存されています。")
                            eichi_plus_shared.stream.output_queue.push((
                                'progress', (None, final_message, make_progress_bar_html(100, translate('処理完了')))
                            ))

                    break
            
            if not self.high_vram:
                self.unload_complete_models()
                flush()
            print(translate("[MEMORY] 処理後のCPUメモリ空き状態: {0:.2f}GB").format(get_main_memory_free_gb()))
            print(translate("[MEMORY] 処理後のGPUメモリ確保状態: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))
        except:
            traceback.print_exc()
            if not self.high_vram:
                self.unload_complete_models()
                flush()
            print(translate("[MEMORY] 処理後のCPUメモリ空き状態: {0:.2f}GB").format(get_main_memory_free_gb()))
            print(translate("[MEMORY] 処理後のGPUメモリ確保状態: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))
    
        eichi_plus_shared.stream.output_queue.push(('end', None))
    