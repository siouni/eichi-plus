import os

import eichi_plus.utils
from eichi_plus.utils import flush, set_hf_home
import eichi_plus.shared.eichi_plus as eichi_plus_shared

from locales.i18n_extended import translate

import time
import torch
import traceback  # デバッグログ出力用
import numpy as np
import einops
from safetensors.torch import save_file, load_file

from yaspin import yaspin

from diffusers_helper.hunyuan import vae_decode_fake
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.gradio.progress_bar import make_progress_bar_html

from eichi_plus.diffusers_helper.memory import (
    cpu, gpu, get_main_memory_free_gb, DynamicSwapInstaller, 
    unload_complete_models,
    offload_model_from_memory_for_storage_preservation,
    move_model_to_device_with_memory_preservation,
    offload_model_from_device_for_memory_preservation,
)

class FramePackI2VSampler:
    def __init__(
            self, high_vram: bool = False, fp8_enabled: bool = False, fp8_path: str = "models/framepack_transfomer_fp8.safetensors", 
            storage_dir: str = "swap_store"
        ):
        self.high_vram: bool = high_vram
        self.storage_dir: str = storage_dir
        self.transformer: HunyuanVideoTransformer3DModelPacked = None
        set_hf_home()
        self.hf_id: str = None
        self.fp8_enabled: bool = fp8_enabled
        self.fp8_path: str = fp8_path
        self.model_loaded = False
    
    def from_pretrained(
            self, hf_id: str = "lllyasviel/FramePackI2V_HY", 
            fp8_enabled: bool = False, fp8_path: str = "models/framepack_transfomer_fp8.safetensors", 
            device=cpu
        ):
        if hf_id is None:
            hf_id = "lllyasviel/FramePackI2V_HY"
        self.hf_id = hf_id
        if fp8_enabled != self.fp8_enabled:
            self.fp8_enabled = fp8_enabled
        self.fp8_path = fp8_path
        print("[MODEL] HunyuanVideoTransformer3DModelPacked Loading...")
        def load_bf16():
            return HunyuanVideoTransformer3DModelPacked.from_pretrained(
                self.hf_id, torch_dtype=torch.bfloat16
            ).to(device, non_blocking=True)
        if self.fp8_enabled:
            from lora_utils.fp8_optimization_utils import check_fp8_support
            has_e4m3, has_e5m2, has_scaled_mm = check_fp8_support()
            if not has_e4m3:
                print(translate("FP8最適化が有効化されていますが、サポートされていません。PyTorch 2.1以上が必要です。"))
                self.transformer = load_bf16()
            else:
                try:
                    if not os.path.isfile(os.path.join(self.fp8_path)):
                        print("[MODEL] FP8量子化済みデータが見つからないので、FP8最適化を実行します。")
                        eichi_plus_shared.stream.output_queue.push(
                            ('progress', (None, '', make_progress_bar_html(0, translate('FP8 Optimization ...'))))
                        )
                        self.transformer = load_bf16()

                        from eichi_plus.model.fp8_optimization_utils import state_dict_with_fp8_optimization
                        state_dict = self.transformer.state_dict()
                        state_dict = state_dict_with_fp8_optimization(state_dict, gpu, weight_hook=None)

                        eichi_plus_shared.stream.output_queue.push(
                            ('progress', (None, '', make_progress_bar_html(0, translate('FP8 Model Saveing ...'))))
                        )
                        with yaspin(text="FP8最適化データ保存処理中...", color="cyan") as spinner:
                            os.makedirs(os.path.dirname(self.fp8_path), exist_ok=True)
                            save_file(state_dict, self.fp8_path)
                            spinner.color = "green"
                            spinner.text = ""
                            spinner.ok("✓ FP8最適化データ保存処理完了")
                        
                        print("[MODEL] 現在のモデルを一度メモリから解放し、FP8版を再読み込みを行います。")
                        eichi_plus_shared.stream.output_queue.push(
                            ('progress', (None, '', make_progress_bar_html(0, translate('Reload Model ...'))))
                        )
                        del self.transformer, state_dict
                        self.transformer = None
                        flush()

                    self.transformer = load_bf16()
                    with yaspin(text="FP8最適化データ読み込み処理中...", color="cyan") as spinner:
                        state_dict = load_file(self.fp8_path)
                        spinner.color = "green"
                        spinner.text = ""
                        spinner.ok("✓ FP8最適化データ読み込み処理完了")

                    from lora_utils.fp8_optimization_utils import apply_fp8_monkey_patch
                    # モンキーパッチの適用
                    print(translate("FP8モンキーパッチを適用しています..."))
                    # use_scaled_mm = has_scaled_mm and has_e5m2
                    use_scaled_mm = False  # 品質が大幅に劣化するので無効化
                    apply_fp8_monkey_patch(self.transformer, state_dict, use_scaled_mm=use_scaled_mm)
                    print(translate("FP8最適化が適用されました！"))
                    # 必要に応じてLoRA、FP8最適化が施された状態辞書を読み込み。assign=Trueで仮想デバイスのテンソルを置換
                    self.transformer.load_state_dict(state_dict, assign=True, strict=True)
                    self.transformer.to(device, non_blocking=True)
                except Exception as e:
                    print(translate("FP8最適化エラー: {0}").format(e))
                    traceback.print_exc()
                    raise e
        else:
            self.transformer = load_bf16()
        self.transformer.high_quality_fp32_output_for_inference = True
        print('transformer.high_quality_fp32_output_for_inference = True')
        self.transformer.eval()
        self.transformer.requires_grad_(False)
        if not self.high_vram:
            print("Install DynamicSwap transformer...")
            DynamicSwapInstaller.install_model(self.transformer, storage_dir=self.storage_dir, device=gpu, non_blocking=True)
        else:
            self.transformer.to(gpu)
        self.model_loaded = True
    
    def to(self, device: torch.device = None, dtype: torch.dtype = None, non_blocking: bool = False):
        if self.transformer is None:
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        self.transformer.to(device, dtype, non_blocking)
    
    def move_model_to_device_with_memory_preservation(self, device=gpu, preserved_gpu_memory=6.0, preserved_cpu_memory=6.0):
        if self.transformer is None:
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        offload_model_from_memory_for_storage_preservation(self.transformer, preserved_memory_gb=preserved_cpu_memory)
        print(translate('Setting transformer memory preservation to: {0} GB').format(preserved_gpu_memory))
        move_model_to_device_with_memory_preservation(self.transformer, target_device=device, preserved_memory_gb=preserved_gpu_memory)
    
    def use_teacache(self, use_teacache, steps):
        if self.transformer is None:
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        if use_teacache:
            self.transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
        else:
            self.transformer.initialize_teacache(enable_teacache=False)
    
    def offload_model_from_device_for_memory_preservation(self, device=gpu, preserved_gpu_memory=8.0, preserved_cpu_memory=6.0):
        if self.transformer is None:
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        offload_model_from_device_for_memory_preservation(
            self.transformer, device, preserved_memory_gb=preserved_gpu_memory, preserved_cpu_memory_gb=preserved_cpu_memory
        )

    def unload_complete_models(self):
        unload_complete_models(
            self.transformer
        )
        del self.transformer
        self.transformer = None
        self.model_loaded = False
    
    def to_dtype_transfomer(self, llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, image_encoder_last_hidden_state):
        if self.transformer is None:
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        llama_vec = llama_vec.to(self.transformer.dtype)
        llama_vec_n = llama_vec_n.to(self.transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(self.transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(self.transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(self.transformer.dtype)
        return llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, image_encoder_last_hidden_state 


    def sampling(
            self, 
            i_section, total_sections, total_generated_latent_frames, steps,
            height, width, num_frames, cfg, gs, rs, rnd,
            current_llama_vec, current_llama_attention_mask, current_clip_l_pooler,
            llama_vec_n, llama_attention_mask_n, clip_l_pooler_n,
            image_encoder_last_hidden_state, latent_indices, 
            clean_latents, clean_latent_indices,
            clean_latents_2x, clean_latent_2x_indices,
            clean_latents_4x, clean_latent_4x_indices,
        ):
        if self.transformer is None:
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        # 処理時間計測の開始
        process_start_time = time.time()
        with yaspin(text="サンプリング処理中...", color="cyan") as spinner:
            try:
                def callback(d):
                    preview = d['denoised']
                    preview = vae_decode_fake(preview)

                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                    if eichi_plus_shared.stream.input_queue.top() == 'end':
                        eichi_plus_shared.stream.output_queue.push(('end', None))
                        raise KeyboardInterrupt('User ends the task.')

                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    hint = translate('Sampling {0}/{1}').format(current_step, steps)
                    # セクション情報を追加（現在のセクション/全セクション）
                    section_info = translate('セクション: {0}/{1}').format(i_section+1, total_sections)
                    desc = f"{section_info} " + translate(
                        '生成フレーム数: {total_generated_latent_frames}, 動画長: {video_length:.2f} 秒 (FPS-30). 動画が生成中です ...'
                    ).format(
                        section_info=section_info, 
                        total_generated_latent_frames=int(max(0, total_generated_latent_frames * 4 - 3)), 
                        video_length=max(0, (total_generated_latent_frames * 4 - 3) / 30)
                    )
                    eichi_plus_shared.stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                    return
                
                with spinner.hidden():
                    generated_latents = sample_hunyuan(
                        transformer=self.transformer,
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
            except:
                traceback.print_exc()
                if not self.high_vram:
                    self.unload_complete_models()
                    flush()
                    spinner.text = ""
                    spinner.color = "yellow"
                    spinner.fail("✘ サンプリング処理に失敗しました。モデルを解放して待機状態に戻ります。")
                    print(translate("[MEMORY] 処理後のCPUメモリ空き状態: {0:.2f}GB").format(get_main_memory_free_gb()))
                    print(translate("[MEMORY] 処理後のGPUメモリ確保状態: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))
                return None
            spinner.text = ""
            spinner.color = "green"
            spinner.ok(f"✓ サンプリング処理完了 {(time.time() - process_start_time):.2f}sec")
        return generated_latents



