import os

from locales.i18n_extended import set_lang, translate
set_lang("ja")

import torch
import gc

def set_hf_home():
    if 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', './hf_download')))
        print(translate("HF_HOMEを設定: {0}").format(os.environ['HF_HOME']))
    else:
        print(translate("既存のHF_HOMEを使用: {0}").format(os.environ['HF_HOME']))

def flush():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

class EventEmitter:
    def __init__(self):
        self._listeners = {}

    def on(self, event_name, callback):
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(callback)

    def off(self, event_name, callback=None):
        if callback is None:
            self._listeners.pop(event_name, None)
        else:
            if event_name in self._listeners:
                self._listeners[event_name] = [cb for cb in self._listeners[event_name] if cb != callback]

    def emit(self, event_name, *args, **kwargs):
        if event_name in self._listeners:
            for callback in self._listeners[event_name]:
                callback(*args, **kwargs)