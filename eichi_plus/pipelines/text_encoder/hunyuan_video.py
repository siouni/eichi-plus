
import eichi_plus.utils
from eichi_plus.utils import flush, set_hf_home

from locales.i18n_extended import translate

import time
import torch
import traceback  # デバッグログ出力用
from transformers import LlamaTokenizerFast, CLIPTokenizer, LlamaModel, CLIPTextModel

from yaspin import yaspin

from diffusers_helper.utils import crop_or_pad_yield_mask
from diffusers_helper.hunyuan import encode_prompt_conds

from eichi_plus.diffusers_helper.memory import (
    get_main_memory_free_gb,
    cpu, gpu, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete,
)

class HunyuanVideoTextEncoder:
    def __init__(self, high_vram: bool = False, storage_dir: str = "swap_store"):
        self.high_vram: bool = high_vram
        self.storage_dir: str = storage_dir

        self.tokenizer: LlamaTokenizerFast = None
        self.tokenizer_2: CLIPTokenizer = None
        self.text_encoder: LlamaModel = None
        self.text_encoder_2: CLIPTextModel = None

        set_hf_home()

        self.hf_id = None

    def from_pretrained(self, hf_id = "hunyuanvideo-community/HunyuanVideo", device=cpu):
        if hf_id is None:
            hf_id = "hunyuanvideo-community/HunyuanVideo"
        self.hf_id = hf_id
        print("[MODEL] LlamaTokenizerFast Loading...")
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            self.hf_id, subfolder='tokenizer'
        )
        print("[MODEL] CLIPTokenizer Loading...")
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.hf_id, subfolder='tokenizer_2'
        )

        print("[MODEL] LlamaModel Loading...")
        self.text_encoder = LlamaModel.from_pretrained(
            self.hf_id, subfolder='text_encoder',
            torch_dtype=torch.float16
        ).to(device, non_blocking=True)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        if not self.high_vram:
            DynamicSwapInstaller.install_model(
                self.text_encoder, storage_dir=self.storage_dir, device=gpu, dtype=torch.float16, non_blocking=True
            )
        else:
            self.text_encoder.to(gpu, non_blocking=True)

        print("[MODEL] CLIPTextModel Loading...")
        self.text_encoder_2 = CLIPTextModel.from_pretrained(
            self.hf_id, subfolder='text_encoder_2',
            torch_dtype=torch.float16
        ).to(device, non_blocking=True)
        self.text_encoder_2.eval()
        self.text_encoder_2.requires_grad_(False)
        if self.high_vram:
            self.text_encoder_2.to(gpu, non_blocking=True)
    
    def to(self, device: torch.device = None, dtype: torch.dtype = None, non_blocking: bool = False):
        self.text_encoder.to(device, dtype, non_blocking)
        self.text_encoder_2.to(device, dtype, non_blocking)
    
    def load_model_as_complete(self, device=gpu):
        if any(v is None for v in (self.tokenizer, self.tokenizer_2, self.text_encoder, self.text_encoder_2)):
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        fake_diffusers_current_device(self.text_encoder, device)
        load_model_as_complete(self.text_encoder_2, device)
    
    def unload_complete_models(self):
        unload_complete_models(
            self.text_encoder, self.text_encoder_2
        )
        del self.text_encoder, self.text_encoder_2
        self.text_encoder = None
        self.text_encoder_2 = None

    
    def encode(self, prompt, n_prompt, cfg, section_map):
        if any(v is None for v in (self.tokenizer, self.tokenizer_2, self.text_encoder, self.text_encoder_2)):
            print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
            self.from_pretrained(self.hf_id, cpu)
        # 処理時間計測の開始
        process_start_time = time.time()
        with yaspin(text="テキストエンコード処理中...", color="cyan") as spinner:
            try:
                llama_vec, clip_l_pooler = encode_prompt_conds(
                    prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2
                )
                if cfg == 1:
                    llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
                else:
                    llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                        n_prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2
                    )
                
                llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
                llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

                # セクションプロンプトを事前にエンコードしておく
                section_prompt_embeddings = {}
                if section_map:
                    with spinner.hidden():
                        print(translate("セクションプロンプトを事前にエンコードしています..."))
                    for sec_num, (_, sec_prompt) in section_map.items():
                        if sec_prompt and sec_prompt.strip():
                            try:
                                # セクションプロンプトをエンコード
                                spinner.text = translate(
                                    "[section_prompt] セクション{0}の専用プロンプトを事前エンコード: {1}..."
                                ).format(sec_num, sec_prompt[:30])
                                # print(translate(
                                #     "[section_prompt] セクション{0}の専用プロンプトを事前エンコード: {1}..."
                                # ).format(sec_num, sec_prompt[:30]))
                                sec_llama_vec, sec_clip_l_pooler = encode_prompt_conds(
                                    sec_prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2
                                )
                                sec_llama_vec, sec_llama_attention_mask = crop_or_pad_yield_mask(sec_llama_vec, length=512)

                                # データ型を明示的にメインプロンプトと合わせる
                                sec_llama_vec = sec_llama_vec.to(dtype=llama_vec.dtype, device=llama_vec.device)
                                sec_clip_l_pooler = sec_clip_l_pooler.to(dtype=clip_l_pooler.dtype, device=clip_l_pooler.device)
                                sec_llama_attention_mask = sec_llama_attention_mask.to(
                                    dtype=llama_attention_mask.dtype, device=llama_attention_mask.device
                                )

                                # 結果を保存
                                section_prompt_embeddings[sec_num] = (sec_llama_vec, sec_clip_l_pooler, sec_llama_attention_mask)
                                spinner.text = translate("[section_prompt] セクション{0}のプロンプトエンコード完了").format(sec_num)
                                # print(translate("[section_prompt] セクション{0}のプロンプトエンコード完了").format(sec_num))
                            except Exception as e:
                                spinner.text = translate("[ERROR] セクション{0}のプロンプトエンコードに失敗: {1}").format(sec_num, e)
                                # print(translate("[ERROR] セクション{0}のプロンプトエンコードに失敗: {1}").format(sec_num, e))
                                traceback.print_exc()
            except:
                traceback.print_exc()
                if not self.high_vram:
                    self.unload_complete_models()
                    flush()
                    spinner.text = ""
                    spinner.color = "yellow"
                    spinner.fail("✘ テキストエンコードに失敗しました。モデルを解放して待機状態に戻ります。")
                    print(translate("[MEMORY] 処理後のCPUメモリ空き状態: {0:.2f}GB").format(get_main_memory_free_gb()))
                    print(translate("[MEMORY] 処理後のGPUメモリ確保状態: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))
                return None, None, None, None, None, None
            
            spinner.text = ""
            spinner.color = "green"
            spinner.ok(f"✓ テキストエンコード処理完了 {(time.time() - process_start_time):.2f}sec")

        return llama_vec, llama_vec_n, clip_l_pooler, clip_l_pooler_n, llama_attention_mask, llama_attention_mask_n, section_prompt_embeddings

    # セクション固有のプロンプト処理を行う関数
    def process_section_prompt(self, i_section, section_map, llama_vec, clip_l_pooler, llama_attention_mask, embeddings_cache=None):
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
                    print(
                        translate(
                            "[section_prompt] セクション{0}の専用プロンプトをキャッシュから取得: {1}..."
                        ).format(i_section, section_prompt[:30])
                    )
                    # キャッシュからデータを取得
                    cached_llama_vec, cached_clip_l_pooler, cached_llama_attention_mask = embeddings_cache[section_num]

                    # データ型を明示的にメインプロンプトと合わせる（2回目のチェック）
                    cached_llama_vec = cached_llama_vec.to(dtype=llama_vec.dtype, device=llama_vec.device)
                    cached_clip_l_pooler = cached_clip_l_pooler.to(dtype=clip_l_pooler.dtype, device=clip_l_pooler.device)
                    cached_llama_attention_mask = cached_llama_attention_mask.to(
                        dtype=llama_attention_mask.dtype, device=llama_attention_mask.device
                    )

                    return cached_llama_vec, cached_clip_l_pooler, cached_llama_attention_mask

                print(translate("[section_prompt] セクション{0}の専用プロンプトを処理: {1}...").format(i_section, section_prompt[:30]))
                if any(v is None for v in (self.tokenizer, self.tokenizer_2, self.text_encoder, self.text_encoder_2)):
                    print("[DEBUG] モデルが読み込まれていません。読み込み処理を実行します。")
                    self.from_pretrained(self.hf_id, cpu)
                try:
                    # プロンプト処理
                    section_llama_vec, section_clip_l_pooler = encode_prompt_conds(
                        section_prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2
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

                    self.unload_complete_models()
                    del self.text_encoder, self.text_encoder_2
                    self.text_encoder = None
                    self.text_encoder_2 = None
                    return section_llama_vec, section_clip_l_pooler, section_llama_attention_mask
                except Exception as e:
                    self.unload_complete_models()
                    del self.text_encoder, self.text_encoder_2
                    self.text_encoder = None
                    self.text_encoder_2 = None
                    print(translate("[ERROR] セクションプロンプト処理エラー: {0}").format(e))

        # 共通プロンプトを使用
        print(translate("[section_prompt] セクション{0}は共通プロンプトを使用します").format(i_section))
        return llama_vec, clip_l_pooler, llama_attention_mask


