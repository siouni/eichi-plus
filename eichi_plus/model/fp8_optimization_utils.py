import os

import torch
from tqdm import tqdm

from lora_utils.fp8_optimization_utils import calculate_fp8_maxval, quantize_tensor_to_fp8

# 国際化対応
from locales.i18n_extended import translate as _

def state_dict_with_fp8_optimization(
    state_dict, device: torch.device, weight_hook: callable = None
) -> dict[str, torch.Tensor]:
    """
    Load state dict from safetensors into the state dict with fp8 optimization if needed.
    """
    # 最適化のターゲットと除外キーを設定
    TARGET_KEYS = ["transformer_blocks", "single_transformer_blocks"]
    EXCLUDE_KEYS = ["norm"]  # Exclude norm layers (e.g., LayerNorm, RMSNorm) from FP8

    # 状態辞書をFP8形式に最適化
    print(_("FP8形式で状態辞書を最適化しています..."))
    optimized_state = optimize_state_dict_with_fp8(
        state_dict, device, TARGET_KEYS, EXCLUDE_KEYS, move_to_device=True, weight_hook=weight_hook
    )
    return optimized_state

def optimize_state_dict_with_fp8(
    state_dict,
    calc_device,
    target_layer_keys=None,
    exclude_layer_keys=None,
    exp_bits=4,
    mantissa_bits=3,
    move_to_device=False,
    weight_hook=None,
):
    """
    モデルの状態辞書内の線形レイヤーの重みをFP8形式に最適化

    Args:
        state_dict (dict): 最適化対象の state_dict
        calc_device (str): テンソルを量子化するデバイス
        target_layer_keys (list, optional): 対象とするレイヤーキーのパターン（Noneの場合はすべての線形レイヤー）
        exclude_layer_keys (list, optional): 除外するレイヤーキーのパターン
        exp_bits (int): 指数部のビット数
        mantissa_bits (int): 仮数部のビット数
        move_to_device (bool): 最適化されたテンソルを計算デバイスに移動するかどうか
        weight_hook (callable, optional): 重みのフック関数（Noneの場合は使用しない）

    Returns:
        dict: FP8最適化された state_dict
    """
    # FP8データ型の選択
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"サポートされていないFP8形式: E{exp_bits}M{mantissa_bits}")

    # FP8の最大値を計算
    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value  # 符号付きFP8

    # 対象キー判定関数
    def is_target_key(key):
        is_target = (target_layer_keys is None or any(p in key for p in target_layer_keys)) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(p in key for p in exclude_layer_keys)
        return is_target and not is_excluded

    optimized_count = 0
    optimized_state = {}

    for key, value in tqdm(state_dict.items(), desc="Optimizing state_dict", unit="param"):
        tensor = value
        # フック適用
        if weight_hook is not None:
            tensor = weight_hook(key, tensor)

        if not is_target_key(key):
            optimized_state[key] = tensor
            continue

        original_device = tensor.device
        original_dtype = tensor.dtype

        # 計算デバイスへ移動
        if calc_device is not None:
            tensor = tensor.to(calc_device)

        # スケールファクタ計算
        scale = torch.max(torch.abs(tensor.view(-1))) / max_value

        # FP8量子化
        quantized_weight, _ = quantize_tensor_to_fp8(
            tensor, scale, exp_bits, mantissa_bits,
            1, max_value, min_value)

        # キー設定
        fp8_key = key
        scale_key = key.replace(".weight", ".scale_weight")

        # FP8型に変換
        quantized_weight = quantized_weight.to(fp8_dtype)

        # 元デバイスに戻す
        if not move_to_device:
            quantized_weight = quantized_weight.to(original_device)

        # スケールテンソル作成
        scale_tensor = torch.tensor([scale], dtype=original_dtype, device=quantized_weight.device)

        optimized_state[fp8_key] = quantized_weight
        optimized_state[scale_key] = scale_tensor

        optimized_count += 1
        # メモリ解放
        if calc_device is not None and optimized_count % 10 == 0:
            torch.cuda.empty_cache()

    print(f"最適化された線形レイヤー数: {optimized_count}")
    return optimized_state