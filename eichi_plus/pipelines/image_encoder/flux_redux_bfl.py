import os

import eichi_plus.utils
from eichi_plus.utils import flush, set_hf_home

from locales.i18n_extended import translate

import time
import torch
import traceback  # デバッグログ出力用

from yaspin import yaspin

from transformers import SiglipVisionModel, SiglipImageProcessor

from diffusers_helper.clip_vision import hf_clip_vision_encode

from eichi_plus.diffusers_helper.memory import (
    cpu, gpu, get_main_memory_free_gb,
    load_model_as_complete, unload_complete_models,
)

class FluxReduxBflImageEncoder:
    def __init__(self, high_vram: bool = False):
        self.high_vram: bool = high_vram

        self.feature_extractor: SiglipImageProcessor = None
        self.image_encoder: SiglipVisionModel = None
    
        set_hf_home()

        self.hf_id = None
    
    def from_pretrained(self, hf_id = "lllyasviel/flux_redux_bfl", device=cpu):
        if hf_id is None:
            hf_id = "lllyasviel/flux_redux_bfl"
        self.hf_id = hf_id

        print("[MODEL] SiglipVisionModel Loading...")
        self.feature_extractor = SiglipImageProcessor.from_pretrained(
            self.hf_id, subfolder='feature_extractor', torch_dtype=torch.float16
        )
        
        print("[MODEL] SiglipVisionModel Loading...")
        self.image_encoder = SiglipVisionModel.from_pretrained(
            self.hf_id, subfolder='image_encoder', torch_dtype=torch.float16
        ).to(device, non_blocking=True)
        self.image_encoder.eval()
        self.image_encoder.requires_grad_(False)
        if self.high_vram:
            self.image_encoder.to(gpu)
    
    def to(self, device: torch.device = None, dtype: torch.dtype = None, non_blocking: bool = False):
        self.image_encoder.to(device, dtype, non_blocking)
    
    def load_model_as_complete(self, device=gpu):
        if any(v is None for v in (self.feature_extractor, self.image_encoder)):
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        load_model_as_complete(self.image_encoder, device)
    
    def unload_complete_models(self):
        unload_complete_models(
            self.image_encoder
        )
        del self.feature_extractor, self.image_encoder
        self.feature_extractor = None
        self.image_encoder = None

    def encode(self, input_image_np):
        if any(v is None for v in (self.feature_extractor, self.image_encoder)):
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        # 処理時間計測の開始
        process_start_time = time.time()
        with yaspin(text="CLIP Visionエンコード処理中...", color="cyan") as spinner:
            try:
                image_encoder_output = hf_clip_vision_encode(input_image_np, self.feature_extractor, self.image_encoder)
                image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
            except:
                traceback.print_exc()
                if not self.high_vram:
                    self.unload_complete_models()
                    flush()
                    spinner.text = ""
                    spinner.color = "yellow"
                    spinner.fail("✘ CLIP Visionエンコードに失敗しました。モデルを解放して待機状態に戻ります。")
                    print(translate("[MEMORY] 処理後のCPUメモリ空き状態: {0:.2f}GB").format(get_main_memory_free_gb()))
                    print(translate("[MEMORY] 処理後のGPUメモリ確保状態: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))
                return None
        
            spinner.text = ""
            spinner.color = "green"
            spinner.ok(f"✓ CLIP Visionエンコード処理完了 {(time.time() - process_start_time):.2f}sec")
        return image_encoder_last_hidden_state