import os

import torch

# Load translations from JSON files
from locales.i18n_extended import set_lang, translate
set_lang("ja")

if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', './hf_download')))
    print(translate("HF_HOMEを設定: {0}").format(os.environ['HF_HOME']))
else:
    print(translate("既存のHF_HOMEを使用: {0}").format(os.environ['HF_HOME']))

# LoRAサポートの確認
has_lora_support = False
try:
    import lora_utils
    has_lora_support = True
    print(translate("LoRAサポートが有効です"))
except ImportError:
    print(translate("LoRAサポートが無効です（lora_utilsモジュールがインストールされていません）"))

from diffusers_helper.memory import gpu, get_cuda_free_memory_gb

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 100

print(translate('Free VRAM {0} GB').format(free_mem_gb))
print(translate('High-VRAM Mode: {0}').format(high_vram))

from diffusers_helper.thread_utils import AsyncStream

stream = AsyncStream()

frame_size_setting = None

temp_dir = "./temp_for_zip_section_info"

# 設定管理モジュールをインポート
from eichi_utils.settings_manager import (
    get_output_folder_path,
    initialize_settings,
    load_settings,
)

# フォルダ構造を先に定義
webui_folder = os.path.dirname(os.path.abspath(os.path.join(__file__, '..', '..')))

# 設定保存用フォルダの設定
settings_folder = os.path.join(webui_folder, 'settings')
os.makedirs(settings_folder, exist_ok=True)

# 設定ファイル初期化
initialize_settings()

# ベースパスを定義
base_path = os.path.dirname(os.path.abspath(os.path.join(__file__, '..', '..')))

# 設定から出力フォルダを取得
app_settings = load_settings()
output_folder_name = app_settings.get('output_folder', 'outputs')
print(translate("設定から出力フォルダを読み込み: {0}").format(output_folder_name))

# 出力フォルダのフルパスを生成
outputs_folder = get_output_folder_path(output_folder_name)
os.makedirs(outputs_folder, exist_ok=True)

batch_stopped = False

copy_metadata = None

transformer_dtype = torch.bfloat16