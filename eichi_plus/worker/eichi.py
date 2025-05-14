import os

import time
import torch
import traceback  # デバッグログ出力用
import safetensors.torch as sf
import numpy as np
from PIL import Image
import einops
import gc

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

import eichi_plus.shared.eichi as eichi_shared

from locales.i18n_extended import translate

from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from transformers import SiglipImageProcessor, SiglipVisionModel

from eichi_utils.png_metadata import (
    embed_metadata_to_png,PROMPT_KEY, SEED_KEY, SECTION_PROMPT_KEY, SECTION_NUMBER_KEY
)
# グローバルなモデル状態管理インスタンスを作成
# 通常モードではuse_f1_model=Falseを指定（デフォルト値なので省略可）
from eichi_utils.transformer_manager import TransformerManager
from eichi_utils.text_encoder_manager import TextEncoderManager
from eichi_utils.video_mode_settings import get_video_seconds
# 設定管理モジュールをインポート
from eichi_utils.settings_manager import (
    get_output_folder_path,
    load_settings,
    save_settings,
)

# from diffusers_helper.memory import cpu, gpu, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.gradio.progress_bar import make_progress_bar_html

from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

from eichi_plus.diffusers_helper.memory import (
    initialize_storage, get_main_memory_free_gb,
    cpu, gpu, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
    offload_model_from_memory_for_storage_preservation,
)

# transformer_manager = None
# text_encoder_manager = None

tokenizer = None
tokenizer_2 = None
vae = None
image_encoder = None
transformer = None
text_encoder = None
text_encoder_2 = None
feature_extractor = None

cache = None

def flush():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

def load_text_encoder(device=gpu, high_vram=False, storage_dir="swap_store") -> LlamaModel:
    text_encoder = LlamaModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder',
        torch_dtype=torch.float16
    ).to(device)
    text_encoder.eval()
    text_encoder.to(dtype=torch.float16)
    text_encoder.requires_grad_(False)
    if not high_vram:
        print("Install DynamicSwap text_encoder...")
        DynamicSwapInstaller.install_model(text_encoder, storage_dir=storage_dir, device=gpu, non_blocking=True)
    else:
        text_encoder.to(gpu)
    return text_encoder

def load_text_encoder_2(device=gpu, high_vram=False) -> CLIPTextModel:
    text_encoder_2 = CLIPTextModel.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2',
        torch_dtype=torch.float16
    ).to(device)
    text_encoder_2.eval()
    text_encoder_2.to(dtype=torch.float16)
    text_encoder_2.requires_grad_(False)
    if high_vram:
        text_encoder_2.to(gpu)
    return text_encoder_2

def load_transformer(device=gpu, high_vram=False, storage_dir="swap_store") -> HunyuanVideoTransformer3DModelPacked:
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
        'lllyasviel/FramePackI2V_HY',
        torch_dtype=torch.bfloat16
    ).to(device)
    transformer.eval()
    transformer.high_quality_fp32_output_for_inference = True
    print('transformer.high_quality_fp32_output_for_inference = True')
    transformer.to(dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    if not high_vram:
        print("Install DynamicSwap transformer...")
        DynamicSwapInstaller.install_model(transformer, storage_dir="swap_store", device=gpu, non_blocking=True)
    else:
        transformer.to(gpu)
    eichi_shared.transformer_dtype = transformer.dtype
    return transformer

storage_dir="swap_store"

def initialize():
    print("Initializing models...")
    # global transformer_manager, text_encoder_manager
    global tokenizer, tokenizer_2, vae, image_encoder, transformer, text_encoder, text_encoder_2, feature_extractor, cache
    # グローバルなモデル状態管理インスタンスを作成
    # 通常モードではuse_f1_model=Falseを指定（デフォルト値なので省略可）
    # transformer_manager = TransformerManager(device=gpu, high_vram_mode=eichi_shared.high_vram, use_f1_model=False)
    # text_encoder_manager = TextEncoderManager(device=gpu, high_vram_mode=eichi_shared.high_vram)
    try:
        initialize_storage(storage_dir)
        tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
        tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
        vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

        # text_encoderとtext_encoder_2の初期化
        # if not text_encoder_manager.ensure_text_encoder_state():
        #     raise Exception(translate("text_encoderとtext_encoder_2の初期化に失敗しました"))
        # text_encoder, text_encoder_2 = text_encoder_manager.get_text_encoders()
        text_encoder = load_text_encoder(cpu, high_vram=eichi_shared.high_vram, storage_dir=storage_dir)
        text_encoder_2 = load_text_encoder_2(cpu, high_vram=eichi_shared.high_vram)

        # transformerの初期化
        # if not transformer_manager.ensure_transformer_state():
        #     raise Exception(translate("transformerの初期化に失敗しました"))
        # transformer = transformer_manager.get_transformer()  # 仮想デバイス上のtransformerを取得
        transformer = load_transformer(cpu, high_vram=eichi_shared.high_vram, storage_dir=storage_dir)

        # 他のモデルの読み込み
        feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
        image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
    except Exception as e:
        print(translate("モデル読み込みエラー: {0}").format(e))
        print(translate("プログラムを終了します..."))
        import sys
        sys.exit(1)

    vae.eval()
    image_encoder.eval()

    if not eichi_shared.high_vram:
        vae.enable_slicing()
        vae.enable_tiling()

    vae.to(dtype=torch.float16)
    image_encoder.to(dtype=torch.float16)

    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)

    if not eichi_shared.high_vram:
        # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
        # DynamicSwapInstaller.install_model(text_encoder, device=gpu)
        # DynamicSwapInstaller.install_model(transformer, device=gpu) # クラスを操作するので仮想デバイス上のtransformerでもOK
        pass
    else:
        image_encoder.to(gpu)
        vae.to(gpu)

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf=16, all_padding_value=1.0, end_frame=None, end_frame_strength=1.0, keep_section_videos=False, lora_files=None, lora_files2=None, lora_scales_text="0.8,0.8", output_dir=None, save_section_frames=False, section_settings=None, use_all_padding=False, use_lora=False, save_tensor_data=False, tensor_data_input=None, fp8_optimization=False, resolution=640, batch_index=None):
    # global transformer_manager, text_encoder_manager
    global tokenizer, tokenizer_2, vae, image_encoder, transformer, text_encoder, text_encoder_2, feature_extractor

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
            current_latent_window_size = 4.5 if eichi_shared.frame_size_setting == "0.5秒 (17フレーム)" else 9
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
        eichi_shared.outputs_folder = get_output_folder_path(output_dir)
        print(translate("出力フォルダを設定: {0}").format(eichi_shared.outputs_folder))

        # フォルダ名が現在の設定と異なる場合は設定ファイルを更新
        if output_dir != eichi_shared.output_folder_name:
            settings = load_settings()
            settings['output_folder'] = output_dir
            if save_settings(settings):
                eichi_shared.output_folder_name = output_dir
                print(translate("出力フォルダ設定を保存しました: {0}").format(output_dir))
    else:
        # デフォルト設定を使用
        outputs_folder = get_output_folder_path(eichi_shared.output_folder_name)
        print(translate("デフォルト出力フォルダを使用: {0}").format(eichi_shared.outputs_folder))

    # フォルダが存在しない場合は作成
    os.makedirs(eichi_shared.outputs_folder, exist_ok=True)



    # 処理時間計測の開始
    process_start_time = time.time()


    # グローバル変数で状態管理しているモデル変数を宣言する
    # global transformer, text_encoder, text_encoder_2

    # text_encoderとtext_encoder_2を確実にロード
    # if not text_encoder_manager.ensure_text_encoder_state():
    #     raise Exception(translate("text_encoderとtext_encoder_2の初期化に失敗しました"))
    # text_encoder, text_encoder_2 = text_encoder_manager.get_text_encoders()
    if text_encoder is None:
        text_encoder = load_text_encoder(cpu, high_vram=eichi_shared.high_vram)
    if text_encoder_2 is None:
        text_encoder_2 = load_text_encoder_2(cpu, high_vram=eichi_shared.high_vram)


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

    eichi_shared.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
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
        section_numbers_sorted = sorted(section_map.keys()) if section_map else []

        def get_section_info(i_section):
            """
            i_section: int
            section_map: {セクション番号: (画像, プロンプト)}
            指定がなければ次のセクション、なければNone
            """
            if not section_map:
                return None, None, None
            # i_section以降で最初に見つかる設定
            for sec in range(i_section, max(section_numbers_sorted)+1):
                if sec in section_map:
                    img, prm = section_map[sec]
                    return sec, img, prm
            return None, None, None

        # セクション固有のプロンプト処理を行う関数
        def process_section_prompt(i_section, section_map, llama_vec, clip_l_pooler, llama_attention_mask, embeddings_cache=None):
            """セクションに固有のプロンプトがあればエンコードまたはキャッシュから取得して返す
            なければメインプロンプトのエンコード結果を返す
            返り値: (llama_vec, clip_l_pooler, llama_attention_mask)
            """
            if not isinstance(llama_vec, torch.Tensor) or not isinstance(llama_attention_mask, torch.Tensor):
                print(translate("[ERROR] メインプロンプトのエンコード結果またはマスクが不正です"))
                return llama_vec, clip_l_pooler, llama_attention_mask

            # embeddings_cacheがNoneの場合は空の辞書で初期化
            embeddings_cache = embeddings_cache or {}

            # セクション固有のプロンプトがあるか確認
            section_info = None
            section_num = None
            if section_map:
                valid_section_nums = [k for k in section_map.keys() if k >= i_section]
                if valid_section_nums:
                    section_num = min(valid_section_nums)
                    section_info = section_map[section_num]

            # セクション固有のプロンプトがあれば使用
            if section_info:
                img, section_prompt = section_info
                if section_prompt and section_prompt.strip():
                    # 事前にエンコードされたプロンプト埋め込みをキャッシュから取得
                    if section_num in embeddings_cache:
                        print(translate("[section_prompt] セクション{0}の専用プロンプトをキャッシュから取得: {1}...").format(i_section, section_prompt[:30]))
                        # キャッシュからデータを取得
                        cached_llama_vec, cached_clip_l_pooler, cached_llama_attention_mask = embeddings_cache[section_num]

                        # データ型を明示的にメインプロンプトと合わせる（2回目のチェック）
                        cached_llama_vec = cached_llama_vec.to(dtype=llama_vec.dtype, device=llama_vec.device)
                        cached_clip_l_pooler = cached_clip_l_pooler.to(dtype=clip_l_pooler.dtype, device=clip_l_pooler.device)
                        cached_llama_attention_mask = cached_llama_attention_mask.to(dtype=llama_attention_mask.dtype, device=llama_attention_mask.device)

                        return cached_llama_vec, cached_clip_l_pooler, cached_llama_attention_mask

                    print(translate("[section_prompt] セクション{0}の専用プロンプトを処理: {1}...").format(i_section, section_prompt[:30]))

                    try:
                        # プロンプト処理
                        section_llama_vec, section_clip_l_pooler = encode_prompt_conds(
                            section_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2
                        )

                        # マスクの作成
                        section_llama_vec, section_llama_attention_mask = crop_or_pad_yield_mask(
                            section_llama_vec, length=512
                        )

                        # データ型を明示的にメインプロンプトと合わせる
                        section_llama_vec = section_llama_vec.to(
                            dtype=llama_vec.dtype, device=llama_vec.device
                        )
                        section_clip_l_pooler = section_clip_l_pooler.to(
                            dtype=clip_l_pooler.dtype, device=clip_l_pooler.device
                        )
                        section_llama_attention_mask = section_llama_attention_mask.to(
                            device=llama_attention_mask.device
                        )

                        return section_llama_vec, section_clip_l_pooler, section_llama_attention_mask
                    except Exception as e:
                        print(translate("[ERROR] セクションプロンプト処理エラー: {0}").format(e))

            # 共通プロンプトを使用
            print(translate("[section_prompt] セクション{0}は共通プロンプトを使用します").format(i_section))
            return llama_vec, clip_l_pooler, llama_attention_mask


        # Clean GPU
        if not eichi_shared.high_vram:
            # モデルをCPUにアンロード
            unload_complete_models(
                image_encoder, vae
            )
            flush()


        # Text encoding

        eichi_shared.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("Text encoding ...")))))

        if not eichi_shared.high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

        # セクションプロンプトを事前にエンコードしておく
        section_prompt_embeddings = {}
        if section_map:
            print(translate("セクションプロンプトを事前にエンコードしています..."))
            for sec_num, (_, sec_prompt) in section_map.items():
                if sec_prompt and sec_prompt.strip():
                    try:
                        # セクションプロンプトをエンコード
                        print(translate("[section_prompt] セクション{0}の専用プロンプトを事前エンコード: {1}...").format(sec_num, sec_prompt[:30]))
                        sec_llama_vec, sec_clip_l_pooler = encode_prompt_conds(sec_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                        sec_llama_vec, sec_llama_attention_mask = crop_or_pad_yield_mask(sec_llama_vec, length=512)

                        # データ型を明示的にメインプロンプトと合わせる
                        sec_llama_vec = sec_llama_vec.to(dtype=llama_vec.dtype, device=llama_vec.device)
                        sec_clip_l_pooler = sec_clip_l_pooler.to(dtype=clip_l_pooler.dtype, device=clip_l_pooler.device)
                        sec_llama_attention_mask = sec_llama_attention_mask.to(dtype=llama_attention_mask.dtype, device=llama_attention_mask.device)

                        # 結果を保存
                        section_prompt_embeddings[sec_num] = (sec_llama_vec, sec_clip_l_pooler, sec_llama_attention_mask)
                        print(translate("[section_prompt] セクション{0}のプロンプトエンコード完了").format(sec_num))
                    except Exception as e:
                        print(translate("[ERROR] セクション{0}のプロンプトエンコードに失敗: {1}").format(sec_num, e))
                        traceback.print_exc()


        # これ以降の処理は text_encoder, text_encoder_2 は不要なので、メモリ解放してしまって構わない
        if not eichi_shared.high_vram:
            text_encoder, text_encoder_2 = None, None
            # text_encoder_manager.dispose_text_encoders()
            flush()


        # テンソルデータのアップロードがあれば読み込み
        uploaded_tensor = None
        if tensor_data_input is not None:
            try:
                tensor_path = tensor_data_input.name
                print(translate("テンソルデータを読み込み: {0}").format(os.path.basename(tensor_path)))
                eichi_shared.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate('Loading tensor data ...')))))

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
                    eichi_shared.stream.output_queue.push(('progress', (None, translate('Tensor data loaded successfully!'), make_progress_bar_html(10, translate('Tensor data loaded successfully!')))))
                else:
                    print(translate("警告: テンソルデータに 'history_latents' キーが見つかりません"))
            except Exception as e:
                print(translate("テンソルデータ読み込みエラー: {0}").format(e))
                traceback.print_exc()

        eichi_shared.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("Image processing ...")))))

        def preprocess_image(img_path_or_array, resolution=640):
            """Pathまたは画像配列を処理して適切なサイズに変換する"""
            print(translate("[DEBUG] preprocess_image: img_path_or_array型 = {0}").format(type(img_path_or_array)))

            if img_path_or_array is None:
                # 画像がない場合は指定解像度の黒い画像を生成
                img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
                height = width = resolution
                return img, img, height, width

            # TensorからNumPyへ変換する必要があれば行う
            if isinstance(img_path_or_array, torch.Tensor):
                img_path_or_array = img_path_or_array.cpu().numpy()

            # Pathの場合はPILで画像を開く
            if isinstance(img_path_or_array, str) and os.path.exists(img_path_or_array):
                # print(translate("[DEBUG] ファイルから画像を読み込み: {0}").format(img_path_or_array))
                img = np.array(Image.open(img_path_or_array).convert('RGB'))
            else:
                # NumPy配列の場合はそのまま使う
                img = img_path_or_array

            H, W, C = img.shape
            # 解像度パラメータを使用してサイズを決定
            height, width = find_nearest_bucket(H, W, resolution=resolution)
            img_np = resize_and_center_crop(img, target_width=width, target_height=height)
            img_pt = torch.from_numpy(img_np).float() / 127.5 - 1
            img_pt = img_pt.permute(2, 0, 1)[None, :, None]
            return img_np, img_pt, height, width

        input_image_np, input_image_pt, height, width = preprocess_image(input_image, resolution=resolution)
        Image.fromarray(input_image_np).save(os.path.join(eichi_shared.outputs_folder, f'{job_id}.png'))
        # 入力画像にメタデータを埋め込んで保存
        initial_image_path = os.path.join(eichi_shared.outputs_folder, f'{job_id}.png')
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

        # VAE encoding

        eichi_shared.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("VAE encoding ...")))))

        if not eichi_shared.high_vram:
            load_model_as_complete(vae, target_device=gpu)

        # アップロードされたテンソルがあっても、常に入力画像から通常のエンコーディングを行う
        # テンソルデータは後で後付けとして使用するために保持しておく
        if uploaded_tensor is not None:
            print(translate("アップロードされたテンソルデータを検出: 動画生成後に後方に結合します"))
            # 入力画像がNoneの場合、テンソルからデコードして表示画像を生成
            if input_image is None:
                try:
                    # テンソルの最初のフレームから画像をデコードして表示用に使用
                    preview_latent = uploaded_tensor[:, :, 0:1, :, :].clone()
                    if preview_latent.device != torch.device('cpu'):
                        preview_latent = preview_latent.cpu()
                    if preview_latent.dtype != torch.float16:
                        preview_latent = preview_latent.to(dtype=torch.float16)

                    decoded_image = vae_decode(preview_latent, vae)
                    decoded_image = (decoded_image[0, :, 0] * 127.5 + 127.5).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
                    # デコードした画像を保存
                    Image.fromarray(decoded_image).save(os.path.join(eichi_shared.outputs_folder, f'{job_id}_tensor_preview.png'))
                    # デコードした画像を入力画像として設定
                    input_image = decoded_image
                    # 前処理用のデータも生成
                    input_image_np, input_image_pt, height, width = preprocess_image(input_image)
                    print(translate("テンソルからデコードした画像を生成しました: {0}x{1}").format(height, width))
                except Exception as e:
                    print(translate("テンソルからのデコード中にエラーが発生しました: {0}").format(e))
                    # デコードに失敗した場合は通常の処理を続行

            # UI上でテンソルデータの情報を表示
            tensor_info = translate("テンソルデータ ({0}フレーム) を検出しました。動画生成後に後方に結合します。").format(uploaded_tensor.shape[2])
            eichi_shared.stream.output_queue.push(('progress', (None, tensor_info, make_progress_bar_html(10, translate('テンソルデータを後方に結合')))))

        # 常に入力画像から通常のエンコーディングを行う
        start_latent = vae_encode(input_image_pt, vae)
        # end_frameも同じタイミングでencode
        if end_frame is not None:
            end_frame_np, end_frame_pt, _, _ = preprocess_image(end_frame, resolution=resolution)
            end_frame_latent = vae_encode(end_frame_pt, vae)
        else:
            end_frame_latent = None

        # create section_latents here
        section_latents = None
        if section_map:
            section_latents = {}
            for sec_num, (img, prm) in section_map.items():
                if img is not None:
                    # 画像をVAE encode
                    img_np, img_pt, _, _ = preprocess_image(img, resolution=resolution)
                    section_latents[sec_num] = vae_encode(img_pt, vae)

        # CLIP Vision

        eichi_shared.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("CLIP Vision encoding ...")))))

        if not eichi_shared.high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        if transformer is None:
            transformer = load_transformer(cpu, high_vram=eichi_shared.high_vram)

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        eichi_shared.stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("Start sampling ...")))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        # latent_window_sizeが4.5の場合は特別に17フレームとする
        if latent_window_size == 4.5:
            num_frames = 17  # 5 * 4 - 3 = 17
        else:
            num_frames = int(latent_window_size * 4 - 3)

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        # ここでlatent_paddingsを再定義していたのが原因だったため、再定義を削除します


        # -------- LoRA 設定 START ---------

        # LoRAの環境変数設定（PYTORCH_CUDA_ALLOC_CONF）
        # if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        #     old_env = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        #     os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        #     print(translate("CUDA環境変数設定: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (元の値: {0})").format(old_env))

        # 次回のtransformer設定を更新
        # current_lora_paths = []
        # current_lora_scales = []
        
        # if use_lora and eichi_shared.has_lora_support:
        #     # LoRAファイルを収集
        #     if lora_files is not None:
        #         if isinstance(lora_files, list):
        #             # 複数のLoRAファイル（将来のGradioバージョン用）
        #             current_lora_paths.extend([file.name for file in lora_files])
        #         else:
        #             # 単一のLoRAファイル
        #             current_lora_paths.append(lora_files.name)
            
        #     # 2つ目のLoRAファイルがあれば追加
        #     if lora_files2 is not None:
        #         if isinstance(lora_files2, list):
        #             # 複数のLoRAファイル（将来のGradioバージョン用）
        #             current_lora_paths.extend([file.name for file in lora_files2])
        #         else:
        #             # 単一のLoRAファイル
        #             current_lora_paths.append(lora_files2.name)
            
        #     # スケール値をテキストから解析
        #     if current_lora_paths:  # LoRAパスがある場合のみ解析
        #         try:
        #             scales_text = lora_scales_text.strip()
        #             if scales_text:
        #                 # カンマ区切りのスケール値を解析
        #                 scales = [float(scale.strip()) for scale in scales_text.split(',')]
        #                 current_lora_scales = scales
        #             else:
        #                 # スケール値が指定されていない場合は全て0.8を使用
        #                 current_lora_scales = [0.8] * len(current_lora_paths)
        #         except Exception as e:
        #             print(translate("LoRAスケール解析エラー: {0}").format(e))
        #             print(translate("デフォルトスケール 0.8 を使用します"))
        #             current_lora_scales = [0.8] * len(current_lora_paths)
                
        #         # スケール値の数がLoRAパスの数と一致しない場合は調整
        #         if len(current_lora_scales) < len(current_lora_paths):
        #             # 足りない分は0.8で埋める
        #             current_lora_scales.extend([0.8] * (len(current_lora_paths) - len(current_lora_scales)))
        #         elif len(current_lora_scales) > len(current_lora_paths):
        #             # 余分は切り捨て
        #             current_lora_scales = current_lora_scales[:len(current_lora_paths)]
        
        # LoRA設定を更新（リロードは行わない）
        # transformer_manager.set_next_settings(
        #     lora_paths=current_lora_paths,
        #     lora_scales=current_lora_scales,
        #     fp8_enabled=fp8_optimization,
        #     high_vram_mode=eichi_shared.high_vram,
        # )

        # -------- LoRA 設定 END ---------

        # セクション処理開始前にtransformerの状態を確認
        # print(translate("\nセクション処理開始前のtransformer状態チェック..."))
        # try:
        #     # transformerの状態を確認し、必要に応じてリロード
        #     if not transformer_manager.ensure_transformer_state():
        #         raise Exception(translate("transformer状態の確認に失敗しました"))

        #     # 最新のtransformerインスタンスを取得
        #     transformer = transformer_manager.get_transformer()
        #     print(translate("transformer状態チェック完了"))
        # except Exception as e:
        #     print(translate("transformer状態チェックエラー: {0}").format(e))
        #     traceback.print_exc()
        #     raise e

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
                    print(translate("パディング値変換: セクション{0}の値を{1}から{2}に変換しました").format(i_section, orig_padding_value, latent_padding))
            else:
                # 通常モードの場合
                is_last_section = latent_padding == 0

            use_end_latent = is_last_section and end_frame is not None
            latent_padding_size = int(latent_padding * latent_window_size)

            # 定義後にログ出力
            padding_info = translate("設定パディング値: {0}").format(all_padding_value) if use_all_padding else translate("パディング値: {0}").format(latent_padding)
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
                    print(translate("[section_latent] section {0}: use section {1} latent (section_map keys: {2})").format(i_section, use_key, list(section_latents.keys())))
                    print(translate("[section_latent] current_latent id: {0}, min: {1:.4f}, max: {2:.4f}, mean: {3:.4f}").format(id(current_latent), current_latent.min().item(), current_latent.max().item(), current_latent.mean().item()))
                else:
                    current_latent = start_latent
                    print(translate("[section_latent] section {0}: use start_latent (no section_latent >= {1})").format(i_section, i_section))
                    print(translate("[section_latent] current_latent id: {0}, min: {1:.4f}, max: {2:.4f}, mean: {3:.4f}").format(id(current_latent), current_latent.min().item(), current_latent.max().item(), current_latent.mean().item()))
            else:
                current_latent = start_latent
                print(translate("[section_latent] section {0}: use start_latent (no section_latents)").format(i_section))
                print(translate("[section_latent] current_latent id: {0}, min: {1:.4f}, max: {2:.4f}, mean: {3:.4f}").format(id(current_latent), current_latent.min().item(), current_latent.max().item(), current_latent.mean().item()))

            if is_first_section and end_frame_latent is not None:
                # EndFrame影響度設定を適用（デフォルトは1.0=通常の影響）
                if end_frame_strength != 1.0:
                    # 影響度を適用した潜在表現を生成
                    # 値が小さいほど影響が弱まるように単純な乗算を使用
                    # end_frame_strength=1.0のときは1.0倍（元の値）
                    # end_frame_strength=0.01のときは0.01倍（影響が非常に弱い）
                    modified_end_frame_latent = end_frame_latent * end_frame_strength
                    print(translate("EndFrame影響度を{0}に設定（最終フレームの影響が{1}倍）").format(f"{end_frame_strength:.2f}", f"{end_frame_strength:.2f}"))
                    history_latents[:, :, 0:1, :, :] = modified_end_frame_latent
                else:
                    # 通常の処理（通常の影響）
                    history_latents[:, :, 0:1, :, :] = end_frame_latent

            if eichi_shared.stream.input_queue.top() == 'end':
                eichi_shared.stream.output_queue.push(('end', None))
                return

            # セクション固有のプロンプトがあれば使用する（事前にエンコードしたキャッシュを使用）
            current_llama_vec, current_clip_l_pooler, current_llama_attention_mask = process_section_prompt(i_section, section_map, llama_vec, clip_l_pooler, llama_attention_mask, section_prompt_embeddings)

            print(translate('latent_padding_size = {0}, is_last_section = {1}').format(latent_padding_size, is_last_section))


            # COMMENTED OUT: セクション処理前のメモリ解放（処理速度向上のため）
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()

            # latent_window_sizeが4.5の場合は特別に5を使用
            effective_window_size = 5 if latent_window_size == 4.5 else int(latent_window_size)
            indices = torch.arange(0, sum([1, latent_padding_size, effective_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, effective_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = current_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            if not eichi_shared.high_vram:
                unload_complete_models()
                offload_model_from_memory_for_storage_preservation(transformer, preserved_memory_gb=2.0)
                # GPUメモリ保存値を明示的に浮動小数点に変換
                preserved_memory = float(gpu_memory_preservation) if gpu_memory_preservation is not None else 6.0
                print(translate('Setting transformer memory preservation to: {0} GB').format(preserved_memory))
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=preserved_memory)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if eichi_shared.stream.input_queue.top() == 'end':
                    eichi_shared.stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = translate('Sampling {0}/{1}').format(current_step, steps)
                # セクション情報を追加（現在のセクション/全セクション）
                section_info = translate('セクション: {0}/{1}').format(i_section+1, total_sections)
                desc = f"{section_info} " + translate('生成フレーム数: {total_generated_latent_frames}, 動画長: {video_length:.2f} 秒 (FPS-30). 動画が生成中です ...').format(section_info=section_info, total_generated_latent_frames=int(max(0, total_generated_latent_frames * 4 - 3)), video_length=max(0, (total_generated_latent_frames * 4 - 3) / 30))
                eichi_shared.stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=current_llama_vec,  # セクションごとのプロンプトを使用
                prompt_embeds_mask=current_llama_attention_mask,  # セクションごとのマスクを使用
                prompt_poolers=current_clip_l_pooler,  # セクションごとのプロンプトを使用
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not eichi_shared.high_vram:
                transformer = None
                flush()
                transformer = load_transformer(cpu, high_vram=eichi_shared.high_vram, storage_dir=storage_dir)
                # offload_model_from_memory_for_storage_preservation(transformer, preserved_memory_gb=2.0)
                # 減圧時に使用するGPUメモリ値も明示的に浮動小数点に設定
                preserved_memory_offload = 8.0  # こちらは固定値のまま
                print(translate('Offloading transformer with memory preservation: {0} GB').format(preserved_memory_offload))
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=preserved_memory_offload)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            # COMMENTED OUT: VAEデコード前のメモリクリア（処理速度向上のため）
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()
            #     print(translate("VAEデコード前メモリ: {memory_allocated:.2f}GB").format(memory_allocated=torch.cuda.memory_allocated()/1024**3))

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                # latent_window_sizeが4.5の場合は特別に5を使用
                if latent_window_size == 4.5:
                    section_latent_frames = 11 if is_last_section else 10  # 5 * 2 + 1 = 11, 5 * 2 = 10
                    overlapped_frames = 17  # 5 * 4 - 3 = 17
                else:
                    section_latent_frames = int(latent_window_size * 2 + 1) if is_last_section else int(latent_window_size * 2)
                    overlapped_frames = int(latent_window_size * 4 - 3)

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            # COMMENTED OUT: 明示的なCPU転送と不要テンソルの削除（処理速度向上のため）
            # if torch.cuda.is_available():
            #     # 必要なデコード後、明示的にキャッシュをクリア
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()

            # 各セクションの最終フレームを静止画として保存（セクション番号付き）
            if save_section_frames and history_pixels is not None:
                try:
                    if i_section == 0 or current_pixels is None:
                        # 最初のセクションは history_pixels の最後
                        last_frame = history_pixels[0, :, -1, :, :]
                    else:
                        # 2セクション目以降は current_pixels の最後
                        last_frame = current_pixels[0, :, -1, :, :]
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
                        frame_path = os.path.join(eichi_shared.outputs_folder, f'{job_id}_{i_section}_end.png')
                        print(translate("[DEBUG] セクション画像パス: {0}").format(frame_path))
                        Image.fromarray(last_frame).save(frame_path)
                        print(translate("[DEBUG] メタデータ埋め込み実行: {0}").format(section_metadata))
                        embed_metadata_to_png(frame_path, section_metadata)
                    else:
                        frame_path = os.path.join(eichi_shared.outputs_folder, f'{job_id}_{i_section}.png')
                        print(translate("[DEBUG] セクション画像パス: {0}").format(frame_path))
                        Image.fromarray(last_frame).save(frame_path)
                        print(translate("[DEBUG] メタデータ埋め込み実行: {0}").format(section_metadata))
                        embed_metadata_to_png(frame_path, section_metadata)

                    print(translate("\u2713 セクション{0}のフレーム画像をメタデータ付きで保存しました").format(i_section))
                except Exception as e:
                    print(translate("[WARN] セクション{0}最終フレーム画像保存時にエラー: {1}").format(i_section, e))

            if not eichi_shared.high_vram:
                unload_complete_models()

            output_filename = os.path.join(eichi_shared.outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(translate('Decoded. Current latent shape {0}; pixel shape {1}').format(real_history_latents.shape, history_pixels.shape))

            # COMMENTED OUT: セクション処理後の明示的なメモリ解放（処理速度向上のため）
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()
            #     import gc
            #     gc.collect()
            #     memory_allocated = torch.cuda.memory_allocated()/1024**3
            #     memory_reserved = torch.cuda.memory_reserved()/1024**3
            #     print(translate("セクション後メモリ状態: 割当={0:.2f}GB, 予約={1:.2f}GB").format(memory_allocated, memory_reserved))

            print(translate("■ セクション{0}の処理完了").format(i_section))
            print(translate("  - 現在の累計フレーム数: {0}フレーム").format(int(max(0, total_generated_latent_frames * 4 - 3))))
            print(translate("  - レンダリング時間: {0}秒").format(f"{max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f}"))
            print(translate("  - 出力ファイル: {0}").format(output_filename))

            eichi_shared.stream.output_queue.push(('file', output_filename))

            if is_last_section:
                combined_output_filename = None
                # 全セクション処理完了後、テンソルデータを後方に結合
                if uploaded_tensor is not None:
                    try:
                        original_frames = real_history_latents.shape[2]  # 元のフレーム数を記録
                        uploaded_frames = uploaded_tensor.shape[2]  # アップロードされたフレーム数

                        print(translate("テンソルデータを後方に結合します: アップロードされたフレーム数 = {uploaded_frames}").format(uploaded_frames=uploaded_frames))
                        # UI上で進捗状況を更新
                        eichi_shared.stream.output_queue.push(('progress', (None, translate("テンソルデータ({uploaded_frames}フレーム)の結合を開始します...").format(uploaded_frames=uploaded_frames), make_progress_bar_html(80, translate('テンソルデータ結合準備')))))

                        # テンソルデータを後方に結合する前に、互換性チェック
                        # デバッグログを追加して詳細を出力
                        print(translate("[DEBUG] テンソルデータの形状: {0}, 生成データの形状: {1}").format(uploaded_tensor.shape, real_history_latents.shape))
                        print(translate("[DEBUG] テンソルデータの型: {0}, 生成データの型: {1}").format(uploaded_tensor.dtype, real_history_latents.dtype))
                        print(translate("[DEBUG] テンソルデータのデバイス: {0}, 生成データのデバイス: {1}").format(uploaded_tensor.device, real_history_latents.device))

                        if uploaded_tensor.shape[3] != real_history_latents.shape[3] or uploaded_tensor.shape[4] != real_history_latents.shape[4]:
                            print(translate("警告: テンソルサイズが異なります: アップロード={0}, 現在の生成={1}").format(uploaded_tensor.shape, real_history_latents.shape))
                            print(translate("テンソルサイズの不一致のため、前方結合をスキップします"))
                            eichi_shared.stream.output_queue.push(('progress', (None, translate("テンソルサイズの不一致のため、前方結合をスキップしました"), make_progress_bar_html(85, translate('互換性エラー')))))
                        else:
                            # デバイスとデータ型を合わせる
                            processed_tensor = uploaded_tensor.clone()
                            if processed_tensor.device != real_history_latents.device:
                                processed_tensor = processed_tensor.to(real_history_latents.device)
                            if processed_tensor.dtype != real_history_latents.dtype:
                                processed_tensor = processed_tensor.to(dtype=real_history_latents.dtype)

                            # 元の動画を品質を保ちつつ保存
                            original_output_filename = os.path.join(eichi_shared.outputs_folder, f'{job_id}_original.mp4')
                            save_bcthw_as_mp4(history_pixels, original_output_filename, fps=30, crf=mp4_crf)
                            print(translate("元の動画を保存しました: {original_output_filename}").format(original_output_filename=original_output_filename))

                            # 元データのコピーを取得
                            combined_history_latents = real_history_latents.clone()
                            combined_history_pixels = history_pixels.clone() if history_pixels is not None else None

                            # 各チャンクの処理前に明示的にメモリ解放
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                print(translate("[MEMORY] チャンク処理前のGPUメモリ確保状態: {memory:.2f}GB").format(memory=torch.cuda.memory_allocated()/1024**3))

                            # VAEをGPUに移動
                            if not eichi_shared.high_vram and vae.device != torch.device('cuda'):
                                print(translate("[SETUP] VAEをGPUに移動: {0} → cuda").format(vae.device))
                                vae.to('cuda')

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
                            print(translate("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(processed_tensor.min().item(), processed_tensor.max().item(), processed_tensor.mean().item()))
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
                                progress_message = translate("テンソルデータ結合中: チャンク {0}/{1} (フレーム {2}-{3}/{4})").format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames)
                                eichi_shared.stream.output_queue.push(('progress', (None, progress_message, make_progress_bar_html(int(80 + chunk_progress * 0.1), translate('テンソルデータ処理中')))))

                                # 現在のチャンクを取得
                                current_chunk = processed_tensor[:, :, chunk_start:chunk_end, :, :]
                                print(translate("チャンク{0}/{1}処理中: フレーム {2}-{3}/{4}").format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames))

                                # メモリ状態を出力
                                if torch.cuda.is_available():
                                    print(translate("[MEMORY] チャンク{0}処理前のGPUメモリ: {1:.2f}GB/{2:.2f}GB").format(chunk_idx+1, torch.cuda.memory_allocated()/1024**3, torch.cuda.get_device_properties(0).total_memory/1024**3))
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
                                    eichi_shared.stream.output_queue.push(('progress', (None, translate("チャンク{0}/{1}のVAEデコード中...").format(chunk_idx+1, num_chunks), make_progress_bar_html(int(80 + chunk_progress * 0.1), translate('デコード処理')))))

                                    # VAEデコード前にテンソル情報を詳しく出力
                                    print(translate("[DEBUG] チャンク{0}のデコード前情報:").format(chunk_idx+1))
                                    print(translate("  - 形状: {0}").format(current_chunk.shape))
                                    print(translate("  - 型: {0}").format(current_chunk.dtype))
                                    print(translate("  - デバイス: {0}").format(current_chunk.device))
                                    print(translate("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(current_chunk.min().item(), current_chunk.max().item(), current_chunk.mean().item()))

                                    # 明示的にデバイスを合わせる
                                    if current_chunk.device != vae.device:
                                        print(translate("  - デバイスをVAEと同じに変更: {0} → {1}").format(current_chunk.device, vae.device))
                                        current_chunk = current_chunk.to(vae.device)

                                    # 型を明示的に合わせる
                                    if current_chunk.dtype != torch.float16:
                                        print(translate("  - データ型をfloat16に変更: {0} → torch.float16").format(current_chunk.dtype))
                                        current_chunk = current_chunk.to(dtype=torch.float16)

                                    # VAEデコード処理
                                    chunk_pixels = vae_decode(current_chunk, vae).cpu()
                                    print(translate("チャンク{0}のVAEデコード完了 (フレーム数: {1})").format(chunk_idx+1, chunk_frames))

                                    # デコード後のピクセルデータ情報を出力
                                    print(translate("[DEBUG] チャンク{0}のデコード結果:").format(chunk_idx+1))
                                    print(translate("  - 形状: {0}").format(chunk_pixels.shape))
                                    print(translate("  - 型: {0}").format(chunk_pixels.dtype))
                                    print(translate("  - デバイス: {0}").format(chunk_pixels.device))
                                    print(translate("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(chunk_pixels.min().item(), chunk_pixels.max().item(), chunk_pixels.mean().item()))

                                    # メモリ使用量を出力
                                    if torch.cuda.is_available():
                                        print(translate("[MEMORY] チャンク{0}デコード後のGPUメモリ: {1:.2f}GB").format(chunk_idx+1, torch.cuda.memory_allocated()/1024**3))

                                    # 結合する
                                    if combined_history_pixels is None:
                                        # 初回のチャンクの場合はそのまま設定
                                        combined_history_pixels = chunk_pixels
                                    else:
                                        # 2回目以降は結合
                                        print(translate("[DEBUG] 結合前の情報:"))
                                        print(translate("  - 既存: {0}, 型: {1}, デバイス: {2}").format(combined_history_pixels.shape, combined_history_pixels.dtype, combined_history_pixels.device))
                                        print(translate("  - 新規: {0}, 型: {1}, デバイス: {2}").format(chunk_pixels.shape, chunk_pixels.dtype, chunk_pixels.device))

                                        # 既存データと新規データで型とデバイスを揃える
                                        if combined_history_pixels.dtype != chunk_pixels.dtype:
                                            print(translate("  - データ型の不一致を修正: {0} → {1}").format(combined_history_pixels.dtype, chunk_pixels.dtype))
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
                                    print(translate("チャンク{0}の結合完了: 現在の組み込みフレーム数 = {1}").format(chunk_idx+1, current_total_frames))

                                    # 中間結果の保存（チャンクごとに保存すると効率が悪いので、最終チャンクのみ保存）
                                    if chunk_idx == num_chunks - 1 or (chunk_idx > 0 and (chunk_idx + 1) % 5 == 0):
                                        # 5チャンクごと、または最後のチャンクで保存
                                        interim_output_filename = os.path.join(eichi_shared.outputs_folder, f'{job_id}_combined_interim_{chunk_idx+1}.mp4')
                                        print(translate("中間結果を保存中: チャンク{0}/{1}").format(chunk_idx+1, num_chunks))
                                        eichi_shared.stream.output_queue.push(('progress', (None, translate("中間結果のMP4変換中... (チャンク{0}/{1})").format(chunk_idx+1, num_chunks), make_progress_bar_html(int(85 + chunk_progress * 0.1), translate('MP4保存中')))))

                                        # MP4として保存
                                        save_bcthw_as_mp4(combined_history_pixels, interim_output_filename, fps=30, crf=mp4_crf)
                                        print(translate("中間結果を保存しました: {0}").format(interim_output_filename))

                                        # 結合した動画をUIに反映するため、出力フラグを立てる
                                        eichi_shared.stream.output_queue.push(('file', interim_output_filename))
                                except Exception as e:
                                    print(translate("チャンク{0}の処理中にエラーが発生しました: {1}").format(chunk_idx+1, e))
                                    traceback.print_exc()

                                    # エラー情報の詳細な出力
                                    print(translate("[ERROR] 詳細エラー情報:"))
                                    print(translate("  - チャンク情報: {0}/{1}, フレーム {2}-{3}/{4}").format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames))
                                    if 'current_chunk' in locals():
                                        print(translate("  - current_chunk: shape={0}, dtype={1}, device={2}").format(current_chunk.shape, current_chunk.dtype, current_chunk.device))
                                    if 'vae' in globals():
                                        print(translate("  - VAE情報: device={0}, dtype={1}").format(vae.device, next(vae.parameters()).dtype))

                                    # GPUメモリ情報
                                    if torch.cuda.is_available():
                                        print(translate("  - GPU使用量: {0:.2f}GB/{1:.2f}GB").format(torch.cuda.memory_allocated()/1024**3, torch.cuda.get_device_properties(0).total_memory/1024**3))

                                    eichi_shared.stream.output_queue.push(('progress', (None, translate("エラー: チャンク{0}の処理に失敗しました - {1}").format(chunk_idx+1, str(e)), make_progress_bar_html(90, translate('エラー')))))
                                    break

                            # 処理完了後に明示的にメモリ解放
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                print(translate("[MEMORY] チャンク処理後のGPUメモリ確保状態: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))

                            # 全チャンクの処理が完了したら、最終的な結合動画を保存
                            if combined_history_pixels is not None:
                                # 結合された最終結果の情報を出力
                                print(translate("[DEBUG] 最終結合結果:"))
                                print(translate("  - 形状: {0}").format(combined_history_pixels.shape))
                                print(translate("  - 型: {0}").format(combined_history_pixels.dtype))
                                print(translate("  - デバイス: {0}").format(combined_history_pixels.device))
                                # 最終結果の保存
                                print(translate("最終結果を保存中: 全{0}チャンク完了").format(num_chunks))
                                eichi_shared.stream.output_queue.push(('progress', (None, translate("結合した動画をMP4に変換中..."), make_progress_bar_html(95, translate('最終MP4変換処理')))))

                                # 最終的な結合ファイル名
                                combined_output_filename = os.path.join(eichi_shared.outputs_folder, f'{job_id}_combined.mp4')

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

                                    for filename in os.listdir(eichi_shared.outputs_folder):
                                        if interim_pattern.match(filename):
                                            interim_path = os.path.join(eichi_shared.outputs_folder, filename)
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
                                        eichi_shared.stream.output_queue.push(('progress', (None, translate("中間ファイルを削除しました: {0}").format(files_str), make_progress_bar_html(97, translate('クリーンアップ完了')))))
                                    else:
                                        print(translate("削除対象の中間ファイルは見つかりませんでした"))
                                except Exception as e:
                                    print(translate("中間ファイル削除中にエラーが発生しました: {0}").format(e))
                                    traceback.print_exc()

                                # 結合した動画をUIに反映するため、出力フラグを立てる
                                eichi_shared.stream.output_queue.push(('file', combined_output_filename))

                                # 結合後の全フレーム数を計算して表示
                                combined_frames = combined_history_pixels.shape[2]
                                combined_size_mb = (combined_history_pixels.element_size() * combined_history_pixels.nelement()) / (1024 * 1024)
                                print(translate("結合完了情報: テンソルデータ({0}フレーム) + 新規動画({1}フレーム) = 合計{2}フレーム").format(uploaded_frames, original_frames, combined_frames))
                                print(translate("結合動画の再生時間: {0:.2f}秒").format(combined_frames / 30))
                                print(translate("データサイズ: {0:.2f} MB（制限無し）").format(combined_size_mb))

                                # UI上で完了メッセージを表示
                                eichi_shared.stream.output_queue.push(('progress', (None, translate("テンソルデータ({0}フレーム)と動画({1}フレーム)の結合が完了しました。\n合計フレーム数: {2}フレーム ({3:.2f}秒) - サイズ制限なし").format(uploaded_frames, original_frames, combined_frames, combined_frames / 30), make_progress_bar_html(100, translate('結合完了')))))
                            else:
                                print(translate("テンソルデータの結合に失敗しました。"))
                                eichi_shared.stream.output_queue.push(('progress', (None, translate("テンソルデータの結合に失敗しました。"), make_progress_bar_html(100, translate('エラー')))))

                            # 正しく結合された動画はすでに生成済みなので、ここでの処理は不要

                            # この部分の処理はすでに上記のチャンク処理で完了しているため不要

                            # real_history_latentsとhistory_pixelsを結合済みのものに更新
                            real_history_latents = combined_history_latents
                            history_pixels = combined_history_pixels

                            # 結合した動画をUIに反映するため、出力フラグを立てる
                            eichi_shared.stream.output_queue.push(('file', combined_output_filename))

                            # 出力ファイル名を更新
                            output_filename = combined_output_filename

                            # 結合後の全フレーム数を計算して表示
                            combined_frames = combined_history_pixels.shape[2]
                            combined_size_mb = (combined_history_pixels.element_size() * combined_history_pixels.nelement()) / (1024 * 1024)
                            print(translate("結合完了情報: テンソルデータ({0}フレーム) + 新規動画({1}フレーム) = 合計{2}フレーム").format(uploaded_frames, original_frames, combined_frames))
                            print(translate("結合動画の再生時間: {0:.2f}秒").format(combined_frames / 30))
                            print(translate("データサイズ: {0:.2f} MB（制限無し）").format(combined_size_mb))

                            # UI上で完了メッセージを表示
                            eichi_shared.stream.output_queue.push(('progress', (None, translate("テンソルデータ({0}フレーム)と動画({1}フレーム)の結合が完了しました。\n合計フレーム数: {2}フレーム ({3:.2f}秒)").format(uploaded_frames, original_frames, combined_frames, combined_frames / 30), make_progress_bar_html(100, translate('結合完了')))))
                    except Exception as e:
                        print(translate("テンソルデータ結合中にエラーが発生しました: {0}").format(e))
                        traceback.print_exc()
                        eichi_shared.stream.output_queue.push(('progress', (None, translate("エラー: テンソルデータ結合に失敗しました - {0}").format(str(e)), make_progress_bar_html(100, translate('エラー')))))

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
                    print(translate("[MEMORY] 処理完了後のメモリクリア: {memory:.2f}GB/{total_memory:.2f}GB").format(memory=torch.cuda.memory_allocated()/1024**3, total_memory=torch.cuda.get_device_properties(0).total_memory/1024**3))

                # テンソルデータの保存処理
                print(translate("[DEBUG] テンソルデータ保存フラグの値: {0}").format(save_tensor_data))
                if save_tensor_data:
                    try:
                        # 結果のテンソルを保存するファイルパス
                        tensor_file_path = os.path.join(eichi_shared.outputs_folder, f'{job_id}.safetensors')

                        # 保存するデータを準備
                        print(translate("=== テンソルデータ保存処理開始 ==="))
                        print(translate("保存対象フレーム数: {frames}").format(frames=real_history_latents.shape[2]))

                        # サイズ制限を完全に撤廃し、全フレームを保存
                        tensor_to_save = real_history_latents.clone().cpu()

                        # テンソルデータの保存サイズの概算
                        tensor_size_mb = (tensor_to_save.element_size() * tensor_to_save.nelement()) / (1024 * 1024)

                        print(translate("テンソルデータを保存中... shape: {shape}, フレーム数: {frames}, サイズ: {size:.2f} MB").format(shape=tensor_to_save.shape, frames=tensor_to_save.shape[2], size=tensor_size_mb))
                        eichi_shared.stream.output_queue.push(('progress', (None, translate('テンソルデータを保存中... ({frames}フレーム)').format(frames=tensor_to_save.shape[2]), make_progress_bar_html(95, translate('テンソルデータの保存')))))

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
                        eichi_shared.stream.output_queue.push(('progress', (None, translate("テンソルデータが保存されました: {path} ({frames}フレーム, {size:.2f} MB)").format(path=os.path.basename(tensor_file_path), frames=tensor_to_save.shape[2], size=tensor_size_mb), make_progress_bar_html(100, translate('処理完了')))))

                        # アップロードされたテンソルデータがあれば、それも結合したものを保存する
                        if tensor_data_input is not None and uploaded_tensor is not None:
                            try:
                                # アップロードされたテンソルデータのファイル名を取得
                                uploaded_tensor_filename = os.path.basename(tensor_data_input.name)
                                tensor_combined_path = os.path.join(eichi_shared.outputs_folder, f'{job_id}_combined_tensors.safetensors')

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
                                    print(translate("結合テンソル情報: 合計{0}フレーム ({1}+{2}), {3:.2f} MB").format(frames, tensor_to_save.shape[2], uploaded_tensor.shape[2], size))
                                    print(translate("=== テンソルデータ結合処理完了 ==="))
                                    eichi_shared.stream.output_queue.push(('progress', (None, translate("テンソルデータ結合が保存されました: 合計{frames}フレーム").format(frames=combined_frames), make_progress_bar_html(100, translate('結合テンソル保存完了')))))
                            except Exception as e:
                                print(translate("テンソルデータ結合保存エラー: {0}").format(e))
                                traceback.print_exc()
                    except Exception as e:
                        print(translate("テンソルデータ保存エラー: {0}").format(e))
                        traceback.print_exc()
                        eichi_shared.stream.output_queue.push(('progress', (None, translate("テンソルデータの保存中にエラーが発生しました。"), make_progress_bar_html(100, translate('処理完了')))))

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
                    completion_message = translate("すべてのセクション({sections}/{total_sections})が完了しました。テンソルデータとの後方結合も完了しました。結合ファイル名: {filename}\n全体の処理時間: {time}").format(sections=sections, total_sections=total_sections, filename=combined_filename_only, time=time_str)
                    # 最終的な出力ファイルを結合したものに変更
                    output_filename = combined_output_filename
                else:
                    # 通常の完了メッセージ
                    completion_message = translate("すべてのセクション({sections}/{total_sections})が完了しました。全体の処理時間: {time}").format(sections=total_sections, total_sections=total_sections, time=time_str)

                eichi_shared.stream.output_queue.push(('progress', (None, completion_message, make_progress_bar_html(100, translate('処理完了')))))

                # 中間ファイルの削除処理
                if not keep_section_videos:
                    # 最終動画のフルパス
                    final_video_path = output_filename
                    final_video_name = os.path.basename(final_video_path)
                    # job_id部分を取得（タイムスタンプ部分）
                    job_id_part = job_id

                    # ディレクトリ内のすべてのファイルを取得
                    files = os.listdir(eichi_shared.outputs_folder)
                    deleted_count = 0

                    for file in files:
                        # 同じjob_idを持つMP4ファイルかチェック
                        # 結合ファイル('combined'を含む)は消さないように保護
                        if file.startswith(job_id_part) and file.endswith('.mp4') \
                           and file != final_video_name \
                           and 'combined' not in file:  # combinedファイルは保護
                            file_path = os.path.join(eichi_shared.outputs_folder, file)
                            try:
                                os.remove(file_path)
                                deleted_count += 1
                                print(translate("[削除] 中間ファイル: {0}").format(file))
                            except Exception as e:
                                print(translate("[エラー] ファイル削除時のエラー {0}: {1}").format(file, e))

                    if deleted_count > 0:
                        print(translate("[済] {0}個の中間ファイルを削除しました。最終ファイルは保存されています: {1}").format(deleted_count, final_video_name))
                        final_message = translate("中間ファイルを削除しました。最終動画と結合動画は保存されています。")
                        eichi_shared.stream.output_queue.push(('progress', (None, final_message, make_progress_bar_html(100, translate('処理完了')))))

                break
        
        # offload_model_from_memory_for_storage(transformer)
        transformer = None
        flush()
        transformer = load_transformer(cpu, high_vram=eichi_shared.high_vram, storage_dir=storage_dir)
        print(translate("[MEMORY] 処理後のCPUメモリ空き状態: {0:.2f}GB").format(get_main_memory_free_gb()))
        print(translate("[MEMORY] 処理後のGPUメモリ確保状態: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))
    except:
        traceback.print_exc()

        if not eichi_shared.high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )
            # offload_model_from_memory_for_storage(transformer)
            transformer = None
            flush()
            transformer = load_transformer(cpu, high_vram=eichi_shared.high_vram, storage_dir=storage_dir)
            print(translate("[MEMORY] 処理後のCPUメモリ空き状態: {0:.2f}GB").format(get_main_memory_free_gb()))
            print(translate("[MEMORY] 処理後のGPUメモリ確保状態: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))

    eichi_shared.stream.output_queue.push(('end', None))
    return
