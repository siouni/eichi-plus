import os

import eichi_plus.utils
from eichi_plus.utils import flush, set_hf_home
from eichi_plus.utils import flush, set_hf_home
import eichi_plus.shared.eichi_plus as eichi_plus_shared

from locales.i18n_extended import translate

import time
import torch
import traceback  # デバッグログ出力用
import numpy as np
from PIL import Image

from yaspin import yaspin

from diffusers import AutoencoderKLHunyuanVideo

from diffusers_helper.hunyuan import vae_decode, vae_encode
from diffusers_helper.utils import resize_and_center_crop, soft_append_bcthw
from diffusers_helper.bucket_tools import find_nearest_bucket
from diffusers_helper.gradio.progress_bar import make_progress_bar_html

from eichi_plus.diffusers_helper.memory import (
    cpu, gpu, get_main_memory_free_gb,
    load_model_as_complete, unload_complete_models,
)

class HunyuanVideoVAE:
    def __init__(self, high_vram: bool = False):
        self.high_vram: bool = high_vram
        self.vae: AutoencoderKLHunyuanVideo = None
        set_hf_home()
        self.hf_id = None
        self.real_history_latents = None
        self.history_pixels = None
        self.current_pixels = None
        self.history_latents = None
    
    def from_pretrained(self, hf_id = "hunyuanvideo-community/HunyuanVideo", device=cpu):
        if hf_id is None:
            hf_id = "hunyuanvideo-community/HunyuanVideo"
        self.hf_id = hf_id
        print("[MODEL] AutoencoderKLHunyuanVideo Loading...")
        self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
            self.hf_id, subfolder='vae', torch_dtype=torch.float16
        ).to(device, non_blocking=True)
        self.vae.eval()
        self.vae.requires_grad_(False)
        if not self.high_vram:
            self.vae.enable_slicing()
            self.vae.enable_tiling()
        else:
            self.vae.to(gpu)
    
    def to(self, device: torch.device = None, dtype: torch.dtype = None, non_blocking: bool = False):
        self.vae.to(device, dtype, non_blocking)
    
    def load_model_as_complete(self, device=gpu):
        if self.vae is None:
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        load_model_as_complete(self.vae, device)
    
    def unload_complete_models(self):
        unload_complete_models(
            self.vae
        )
        del self.vae
        self.vae = None
    
    def uploaded_tensor(
        self, input_image, input_image_np, input_image_pt, uploaded_tensor, outputs_folder, job_id, section_map,
        resolution=640, end_frame=None
    ):
        if self.vae is None:
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        try:
            # アップロードされたテンソルがあっても、常に入力画像から通常のエンコーディングを行う
            # テンソルデータは後で後付けとして使用するために保持しておく
            if uploaded_tensor is not None:
                # 処理時間計測の開始
                process_start_time = time.time()
                print(translate("アップロードされたテンソルデータを検出: 動画生成後に後方に結合します"))
                with yaspin(text="アップロードテンソル処理中...", color="cyan") as spinner:
                    # 入力画像がNoneの場合、テンソルからデコードして表示画像を生成
                    if input_image is None:
                        try:
                            # テンソルの最初のフレームから画像をデコードして表示用に使用
                            preview_latent = uploaded_tensor[:, :, 0:1, :, :].clone()
                            if preview_latent.device != torch.device('cpu'):
                                preview_latent = preview_latent.cpu()
                            if preview_latent.dtype != torch.float16:
                                preview_latent = preview_latent.to(dtype=torch.float16)

                            decoded_image = self.vae.decode(preview_latent)
                            decoded_image = (decoded_image[0, :, 0] * 127.5 + 127.5).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
                            # デコードした画像を保存
                            Image.fromarray(decoded_image).save(os.path.join(outputs_folder, f'{job_id}_tensor_preview.png'))
                            # デコードした画像を入力画像として設定
                            input_image = decoded_image
                            # 前処理用のデータも生成
                            input_image_np, input_image_pt, height, width = self.preprocess_image(input_image, resolution)
                            with spinner.hidden():
                                print(translate("テンソルからデコードした画像を生成しました: {0}x{1}").format(height, width))
                        except Exception as e:
                            with spinner.hidden():
                                print(translate("テンソルからのデコード中にエラーが発生しました: {0}").format(e))
                            # デコードに失敗した場合は通常の処理を続行

                    # UI上でテンソルデータの情報を表示
                    tensor_info = translate("テンソルデータ ({0}フレーム) を検出しました。動画生成後に後方に結合します。").format(uploaded_tensor.shape[2])
                    eichi_plus_shared.stream.output_queue.push(
                        ('progress', (None, tensor_info, make_progress_bar_html(10, translate('テンソルデータを後方に結合'))))
                    )
                spinner.text = ""
                spinner.color = "green"
                spinner.ok(f"✓ アップロードテンソル処理完了 {(time.time() - process_start_time):.2f}sec")

            # 常に入力画像から通常のエンコーディングを行う
            start_latent = self.encode(input_image_pt)
            # end_frameも同じタイミングでencode
            if end_frame is not None:
                end_frame_np, end_frame_pt, _, _ = self.preprocess_image(end_frame, resolution=resolution)
                end_frame_latent = self.encode(end_frame_pt)
            else:
                end_frame_latent = None

            # create section_latents here
            section_latents = None
            if section_map:
                section_latents = {}
                for sec_num, (img, prm) in section_map.items():
                    if img is not None:
                        # 画像をVAE encode
                        img_np, img_pt, _, _ = self.preprocess_image(img, resolution=resolution)
                        section_latents[sec_num] = self.encode(img_pt)
        except:
            traceback.print_exc()
            if not self.high_vram:
                self.unload_complete_models()
                flush()
                spinner.text = ""
                spinner.color = "yellow"
                spinner.fail("✘ VAEエンコードに失敗しました。モデルを解放して待機状態に戻ります。")
                print(translate("[MEMORY] 処理後のCPUメモリ空き状態: {0:.2f}GB").format(get_main_memory_free_gb()))
                print(translate("[MEMORY] 処理後のGPUメモリ確保状態: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))
            return None, None, None, None
        
        return input_image_np, start_latent, section_latents, end_frame_latent

    def encode(self, pixels):
        if self.vae is None:
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        # 処理時間計測の開始
        process_start_time = time.time()
        with yaspin(text="VAEエンコード処理中...", color="cyan") as spinner:
            latents = vae_encode(pixels, self.vae)
            spinner.text = ""
            spinner.color = "green"
            spinner.ok(f"✓ VAEエンコード処理完了 {(time.time() - process_start_time):.2f}sec")
        return latents

    def decode(self, latents):
        if self.vae is None:
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        # 処理時間計測の開始
        process_start_time = time.time()
        with yaspin(text="VAEデコード処理中...", color="cyan") as spinner:
            pixels = vae_decode(latents, self.vae)
            spinner.text = ""
            spinner.color = "green"
            spinner.ok(f"✓ VAEデコード処理完了 {(time.time() - process_start_time):.2f}sec")
        return pixels

    def preprocess_image(self, img_path_or_array, resolution=640):
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
    
    def init_history_pixels(self, height, width):
        self.real_history_latents = None
        self.history_pixels = None
        self.current_pixels = None
        self.history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()

    def add_history_pixels(self, generated_latents, total_generated_latent_frames, latent_window_size, is_last_section):
        if self.vae is None:
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        # 処理時間計測の開始
        process_start_time = time.time()
        with yaspin(text="history_pixels追加中...", color="cyan") as spinner:
            self.history_latents = torch.cat([generated_latents.to(self.history_latents), self.history_latents], dim=2)

            self.real_history_latents = self.history_latents[:, :, :total_generated_latent_frames, :, :]
            if self.history_pixels is None:
                with spinner.hidden():
                    self.history_pixels = self.decode(self.real_history_latents).cpu()
            else:
                # latent_window_sizeが4.5の場合は特別に5を使用
                if latent_window_size == 4.5:
                    section_latent_frames = 11 if is_last_section else 10  # 5 * 2 + 1 = 11, 5 * 2 = 10
                    overlapped_frames = 17  # 5 * 4 - 3 = 17
                else:
                    section_latent_frames = int(latent_window_size * 2 + 1) if is_last_section else int(latent_window_size * 2)
                    overlapped_frames = int(latent_window_size * 4 - 3)
                with spinner.hidden():
                    self.current_pixels = self.decode(self.real_history_latents[:, :, :section_latent_frames]).cpu()
                self.history_pixels = soft_append_bcthw(self.current_pixels, self.history_pixels, overlapped_frames)
            spinner.text = ""
            spinner.color = "green"
            spinner.ok(f"✓ history_pixels追加完了 {(time.time() - process_start_time):.2f}sec")