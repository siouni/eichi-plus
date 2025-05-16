# worker/eichi_plus.py

import torch

from eichi_plus.shared.eichi import high_vram, frame_size_setting

from eichi_plus.pipelines.pipeline_eichi import EichiPipline

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf=16, all_padding_value=1.0, end_frame=None, end_frame_strength=1.0, keep_section_videos=False, lora_files=None, lora_files2=None, lora_scales_text="0.8,0.8", output_dir=None, save_section_frames=False, section_settings=None, use_all_padding=False, use_lora=False, save_tensor_data=False, tensor_data_input=None, fp8_optimization=False, resolution=640, batch_index=None):

    pipeline = EichiPipline(high_vram)
    pipeline(
        input_image, prompt, n_prompt, seed, steps, cfg, gs, rs, 
        gpu_memory_preservation, use_teacache, 
        section_settings, total_second_length, latent_window_size,
        frame_size_setting,
        mp4_crf,
        use_all_padding, all_padding_value,
        end_frame, end_frame_strength, 
        keep_section_videos, 
        save_tensor_data, tensor_data_input,
        output_dir, save_section_frames, 
        batch_index,
        resolution,
    )