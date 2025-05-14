# By lllyasviel


import torch
# Add by siouni
import os
import gc
import psutil
from yaspin import yaspin
from safetensors.torch import save_file, load_file
import shutil

cpu = torch.device('cpu')
gpu = torch.device(f'cuda:{torch.cuda.current_device()}')
# Add by siouni
storage = "storage"

gpu_complete_modules = []

# Add by siouni
def initialize_storage(storage_dir: str = "swap_store") -> str:
    """ストレージの初期化（ディレクトリ作成＆中身クリア）"""
    # ディレクトリが存在しない場合は作成、存在する場合は中身を削除
    if storage_dir is None:
        storage_dir = "swap_store"
    if os.path.exists(storage_dir):
        # 中身を全削除
        for entry in os.listdir(storage_dir):
            path = os.path.join(storage_dir, entry)
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    else:
        os.makedirs(storage_dir, exist_ok=True)
    print(f"Storage initialized and cleared at {storage_dir}")
    return storage_dir

# Add by siouni
def save_storage(path: str, data: dict[str, torch.Tensor]):
    to_save = {k: v for k, v in data.items() if v is not None}
    if to_save:
        save_file(to_save, path)

# Add by siouni
def load_storage(path: str):
    if os.path.exists(path):
        data = load_file(path)
        return data
    return None

# Add by siouni
def swap_mmap(path: str, data: dict[str, torch.Tensor]):
    save_storage(path, data)
    if os.path.exists(path):
        data = load_storage(path)
        os.remove(path)
    else:
        data = None
    return data

# Chage by siouni
class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, module_name: str, storage_dir: str = "swap_store", **kwargs):
        original_class = module.__class__
        module.__dict__['forge_backup_original_class'] = original_class
        module.__dict__['device'] = cpu
        ext = "safetensors"
        parameters_path = f"{storage_dir}/{module_name}_parameters.{ext}"
        buffers_path = f"{storage_dir}/{module_name}_buffers.{ext}"

        def hacked_get_attr(self, name: str):
            params_dict = object.__getattribute__(self, "_parameters")
            if name in params_dict:
                p = params_dict[name]
                if p is None:
                    return None
                if p.__class__ == torch.nn.Parameter:
                    return torch.nn.Parameter(p.to(**kwargs), requires_grad=p.requires_grad)
                else:
                    return p.to(**kwargs)
            buffers_dict = object.__getattribute__(self, "_buffers")
            if name in buffers_dict:
                b = buffers_dict[name]
                if b is None:
                    return None
                else:
                    return b.to(**kwargs)
            return super(original_class, self).__getattr__(name)

        def custom_to(self, device=None, **to_kwargs):
            # 退避処理
            if device == storage:
                params_dict = object.__getattribute__(self, "_parameters")
                data = swap_mmap(parameters_path, params_dict)
                if data is not None:
                    params_dict.update(data)
                buffers_dict = object.__getattribute__(self, "_buffers")
                data = swap_mmap(buffers_path, buffers_dict)
                if data is not None:
                    buffers_dict.update(data)
                module.__dict__['device'] = storage
                return self
            # 通常の .to 動作
            if device == storage:
                device = cpu
            module.__dict__['device'] = device
            return super(original_class, self).to(device, **to_kwargs)

        # クラスを動的に差し替え
        module.__class__ = type(
            f"DiskCacheSwap_{original_class.__name__}",
            (original_class,),
            {"__getattr__": hacked_get_attr, "to": custom_to},
        )

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if 'forge_backup_original_class' in module.__dict__:
            module.__class__ = module.__dict__.pop('forge_backup_original_class')

    @staticmethod
    def install_model(model: torch.nn.Module, storage_dir: str = "swap_store", **kwargs):
        os.makedirs(storage_dir, exist_ok=True)
        for module_name, m in model.named_modules():
            name = model.__class__.__name__ + "-" + m.__class__.__name__ + "-" + module_name
            DynamicSwapInstaller._install_module(m, name, storage_dir=storage_dir, **kwargs)

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for m in model.modules():
            DynamicSwapInstaller._uninstall_module(m)


def fake_diffusers_current_device(model: torch.nn.Module, target_device: torch.device):
    # print(f'Fake diffusers current device: {model.__class__.__name__} to {target_device}')
    with yaspin(text=f'Fake diffusers current device: {model.__class__.__name__} to {target_device}', color="cyan") as spinner:
        if hasattr(model, 'scale_shift_table'):
            model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
            torch.cuda.empty_cache()
            gc.collect()
            target_device_free_memory_gb = get_cuda_free_memory_gb(target_device)
            cpu_free_memory_gb = get_main_memory_free_gb()
            spinner.text = ""
            spinner.color = "green"
            spinner.ok(f"✓ Moved Fake diffusers current device: {model.__class__.__name__} to {target_device}")
            print(f"[MEMORY] VRAM FREE MEMORY: {target_device_free_memory_gb:.2f} GB")
            print(f"[MEMORY] RAM FREE MEMORY: {cpu_free_memory_gb:.2f} GB")
            return

        for k, p in model.named_modules():
            if hasattr(p, 'weight'):
                p.to(target_device)
                torch.cuda.empty_cache()
                gc.collect()
                target_device_free_memory_gb = get_cuda_free_memory_gb(target_device)
                cpu_free_memory_gb = get_main_memory_free_gb()
                spinner.text = ""
                spinner.color = "green"
                spinner.ok(f"✓ Moved Fake diffusers current device: {model.__class__.__name__} to {target_device}")
                print(f"[MEMORY] VRAM FREE MEMORY: {target_device_free_memory_gb:.2f} GB")
                print(f"[MEMORY] RAM FREE MEMORY: {cpu_free_memory_gb:.2f} GB")
                return


def get_cuda_free_memory_gb(device=None):
    if device is None:
        device = gpu

    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats['active_bytes.all.current']
    bytes_reserved = memory_stats['reserved_bytes.all.current']
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / (1024 ** 3)

# Add by siouni
def get_main_memory_free_gb():
    """実メモリ（物理メモリ）の未使用空き容量をGB単位で取得（仮想メモリ除外）"""
    memory_info = psutil.virtual_memory()
    free_memory_bytes = memory_info.free  # 未使用の物理メモリ（キャッシュ/バッファ除く）
    return free_memory_bytes / (1024 ** 3)

# Change by siouni
def move_model_to_device_with_memory_preservation(model, target_device, preserved_memory_gb=6.0):
    # print(f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB')

    with yaspin(
        text=f'Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB', 
        color="cyan"
    ) as spinner:
        for m in model.modules():
            target_device_free_memory_gb = get_cuda_free_memory_gb(target_device)
            cpu_free_memory_gb = get_main_memory_free_gb()
            spinner.text = f'Moving... VRAM FREE MEMORY: {target_device_free_memory_gb:.2f} GB / RAM FREE MEMORY: {cpu_free_memory_gb:.2f} GB'
            if target_device_free_memory_gb <= preserved_memory_gb:
                torch.cuda.empty_cache()
                gc.collect()
                target_device_free_memory_gb = get_cuda_free_memory_gb(target_device)
                cpu_free_memory_gb = get_main_memory_free_gb()
                spinner.text = ""
                spinner.color = "green"
                spinner.ok(f"✓ Moved {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB")
                print(f"[MEMORY] VRAM FREE MEMORY: {target_device_free_memory_gb:.2f} GB")
                print(f"[MEMORY] RAM FREE MEMORY: {cpu_free_memory_gb:.2f} GB")
                return

            if hasattr(m, 'weight'):
                # move to target device
                m.to(device=target_device)
                torch.cuda.empty_cache()
                gc.collect()

        model.to(device=target_device)
        torch.cuda.empty_cache()
        gc.collect()
        target_device_free_memory_gb = get_cuda_free_memory_gb(target_device)
        cpu_free_memory_gb = get_main_memory_free_gb()
        spinner.text = ""
        spinner.color = "green"
        spinner.ok(f"✓ Moved {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB")
        print(f"[MEMORY] VRAM FREE MEMORY: {target_device_free_memory_gb:.2f} GB")
        print(f"[MEMORY] RAM FREE MEMORY: {cpu_free_memory_gb:.2f} GB")
    return

# Change by siouni
def offload_model_from_device_for_memory_preservation(model, target_device, preserved_memory_gb=8.0, preserved_cpu_memory_gb=2.0):
    # print(f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB')

    with yaspin(
        text=f'Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB', 
        color="cyan"
    ) as spinner:
        for m in model.modules():
            target_device_free_memory_gb = get_cuda_free_memory_gb(target_device)
            cpu_free_memory_gb = get_main_memory_free_gb()
            spinner.text = f'Offloading... VRAM FREE MEMORY: {target_device_free_memory_gb:.2f} GB / RAM FREE MEMORY: {cpu_free_memory_gb:.2f} GB'
            if target_device_free_memory_gb >= preserved_memory_gb:
                torch.cuda.empty_cache()
                gc.collect()
                target_device_free_memory_gb = get_cuda_free_memory_gb(target_device)
                cpu_free_memory_gb = get_main_memory_free_gb()
                spinner.text = ""
                spinner.color = "green"
                spinner.ok(f"✓ Offloaded {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB")
                print(f"[MEMORY] VRAM FREE MEMORY: {target_device_free_memory_gb:.2f} GB")
                print(f"[MEMORY] RAM FREE MEMORY: {cpu_free_memory_gb:.2f} GB")
                return

            if hasattr(m, 'weight'):
                if preserved_cpu_memory_gb >= cpu_free_memory_gb:
                    m.to(device=storage)
                else:
                    m.to(device=cpu)
                torch.cuda.empty_cache()
                gc.collect()

        model.to(device=storage)
        torch.cuda.empty_cache()
        gc.collect()
        target_device_free_memory_gb = get_cuda_free_memory_gb(target_device)
        cpu_free_memory_gb = get_main_memory_free_gb()
        spinner.text = ""
        spinner.color = "green"
        spinner.ok(f"✓ Offloaded {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB")
        print(f"[MEMORY] VRAM FREE MEMORY: {target_device_free_memory_gb:.2f} GB")
        print(f"[MEMORY] RAM FREE MEMORY: {cpu_free_memory_gb:.2f} GB")
    return

# Add by siouni
def offload_model_from_memory_for_storage_preservation(model, preserved_memory_gb=2.0):
    # print(f'Offloading {model.__class__.__name__} from cpu to storage with preserve memory: {preserved_memory_gb} GB')
    
    with yaspin(
        text=f'Offloading {model.__class__.__name__} from cpu to storage with preserve memory: {preserved_memory_gb} GB',
        color="cyan"
    ) as spinner:
        target_device_free_memory_gb = _offload_model_from_memory_for_storage_preservation(
            model, preserved_memory_gb=preserved_memory_gb, spinner=spinner
        )
        print(f"[MEMORY] RAM FREE MEMORY: {target_device_free_memory_gb:.2f} GB")
    return

def _offload_model_from_memory_for_storage_preservation(model, preserved_memory_gb=2.0, spinner=None):
    for m in model.modules():
        if m.device == cpu:
            target_device_free_memory_gb = get_main_memory_free_gb()
            if spinner:
                spinner.text = f'Offloading... RAM FREE MEMORY: {target_device_free_memory_gb:.2f} GB'
            if target_device_free_memory_gb >= preserved_memory_gb:
                torch.cuda.empty_cache()
                gc.collect()
                target_device_free_memory_gb = get_main_memory_free_gb()
                if spinner:
                    spinner.text = ""
                    spinner.color = "green"
                    spinner.ok(f"✓ Offloaded {model.__class__.__name__} to storage with preserved memory: {preserved_memory_gb} GB")
                return target_device_free_memory_gb

            if hasattr(m, 'weight'):
                m.to(device=storage)
                torch.cuda.empty_cache()
                gc.collect()

    model.to(device=storage)
    torch.cuda.empty_cache()
    gc.collect()
    target_device_free_memory_gb = get_main_memory_free_gb()
    if spinner:
        spinner.text = ""
        spinner.color = "green"
        spinner.ok(f"✓ Offloaded {model.__class__.__name__} to storage with preserved memory: {preserved_memory_gb} GB")
    return target_device_free_memory_gb

def offload_model_from_memory_for_storage(model):
    # print(f'Offloading {model.__class__.__name__} from cpu to storage')
    
    with yaspin(text=f'Offloading {model.__class__.__name__} from cpu to storage', color="cyan") as spinner:
        for m in model.modules():
            target_device_free_memory_gb = get_main_memory_free_gb()
            spinner.text = f'Offloading... RAM FREE MEMORY: {target_device_free_memory_gb:.2f} GB'
            if hasattr(m, 'weight'):
                m.to(device=storage)
                torch.cuda.empty_cache()
                gc.collect()
        model.to(device=storage)
        torch.cuda.empty_cache()
        gc.collect()
        target_device_free_memory_gb = get_main_memory_free_gb()
        spinner.text = ""
        spinner.color = "green"
        spinner.ok(f"✓ Offloaded {model.__class__.__name__} to storage")
        print(f"[MEMORY] RAM FREE MEMORY: {target_device_free_memory_gb:.2f} GB")
    return

def unload_complete_models(*args):
    for m in gpu_complete_modules + list(args):
        if m is None:
            continue
        m.to(device=cpu)
        print(f'Unloaded {m.__class__.__name__} as complete.')

    gpu_complete_modules.clear()
    torch.cuda.empty_cache()
    return


def load_model_as_complete(model, target_device, unload=True):
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    print(f'Loaded {model.__class__.__name__} to {target_device} as complete.')

    gpu_complete_modules.append(model)
    return
