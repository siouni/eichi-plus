# gradio/eichi_plus/components/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Any
from collections.abc import Generator

import gradio as gr

from eichi_plus.utils import EventEmitter

@dataclass
class BaseComponentInput(ABC):
    @abstractmethod
    def validate(self) -> bool:
        pass

@dataclass
class BaseComponentOutput(ABC):
    @abstractmethod
    def validate(self) -> bool:
        pass

class BaseComponent(ABC):
    def __init__(self, emitter: EventEmitter):
        self.css: str = None
        self.input: BaseComponentOutput = None
        self.output: BaseComponentInput = None
        self.ui: gr.Blocks = None
        self.event_handlers: dict[str, Callable[..., Any]] = {}
        self.emitter: EventEmitter = emitter
    
    @abstractmethod
    def stream_input(self, data: Any) -> None:
        """
        ストリームに入力するメソッド。
        実装クラスで処理を書く。
        """
        pass

    @abstractmethod
    def stream_output(self) -> Generator[Any, None, None]:
        """
        ストリームの出力を返すジェネレーター。
        逐次的に出力データをyieldする。
        """
        yield  # 抽象的にyieldだけ書いておく

    @abstractmethod
    def _build_template(self, css: str) -> gr.Blocks:
        """
        UI構造を構築し返す。Vueのtemplateに相当。
        ここでgr.BlocksやRow, ColumnなどのUIコンポーネントを構成する。
        """
        pass

    @abstractmethod
    def _setup_logic(self):
        """
        イベント登録や動的処理をセットアップする。Vueのscriptに相当。
        """
        pass

    @abstractmethod
    def _get_css(self) -> str:
        """
        スタイル（CSS）を返す。Vueのstyleに相当。
        必要に応じてoverride可能。
        """
        return ""

    def __call__(self):
        """
        UI構築とロジックセットアップをまとめて呼ぶ。
        """
        css = self._get_css()
        self._setup_logic()
        self.ui = self._build_template(css)
        return self.ui

