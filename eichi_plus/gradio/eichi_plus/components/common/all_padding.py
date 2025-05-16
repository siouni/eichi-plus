# gradio/eichi_plus/components/common/section_frame_size.py
import os

from dataclasses import dataclass, field
from typing import Callable, Any
from collections.abc import Generator

import gradio as gr

import eichi_plus.shared.eichi_plus as eichi_plus_shared

from locales.i18n_extended import translate

from eichi_plus.utils import EventEmitter

from eichi_plus.gradio.eichi_plus.components.base import BaseComponent, BaseComponentOutput

@dataclass
class AllPaddingComponentOutput(BaseComponentOutput):
    use_all_padding: bool = False
    all_padding_value: float = 1.0
    def validate(self) -> bool:
        return True

class AllPaddingComponent(BaseComponent):
    def __init__(self, emitter: EventEmitter):
        super().__init__(emitter)

        self.use_all_padding: gr.Checkbox = None
        self.all_padding_value: gr.Slider = None

        self.output: AllPaddingComponentOutput = AllPaddingComponentOutput()

    def stream_input(self, data: Any) -> None:
        pass

    def stream_output(self) -> Generator[Any, None, None]:
        pass

    def _build_template(self, css: str) -> gr.Blocks:
        """
        UI構造を構築し返す。Vueのtemplateに相当。
        ここでgr.BlocksやRow, ColumnなどのUIコンポーネントを構成する。
        """
        block = gr.Blocks(css=css).queue()
        with block:
            self.use_all_padding = gr.Checkbox(
                label=translate("オールパディング"),
                value=False,
                info=translate("数値が小さいほど直前の絵への影響度が下がり動きが増える"),
                elem_id="all_padding_checkbox"
            )
            self.all_padding_value = gr.Slider(
                label=translate("パディング値"),
                minimum=0.2,
                maximum=3,
                value=1,
                step=0.1,
                info=translate("すべてのセクションに適用するパディング値（0.2〜3の整数）"),
                visible=False
            )
            
            self.use_all_padding.change(
                fn=self.event_handlers["on_use_allpading_change"],
                inputs=[self.use_all_padding],
                outputs=[self.all_padding_value]
            )
            self.all_padding_value.change(
                fn=self.event_handlers["on_all_padding_value_change"],
                inputs=[self.all_padding_value],
            )

    def _setup_logic(self):
        """
        イベント登録や動的処理をセットアップする。Vueのscriptに相当。
        """
        def on_use_allpading_change(use_all_padding_value):
            self.output.use_all_padding = use_all_padding_value
            return gr.update(visible=use_all_padding_value)
        
        def on_all_padding_value_change(all_padding_value_value):
            self.output.all_padding_value = all_padding_value_value
        
        self.event_handlers["on_use_allpading_change"] = on_use_allpading_change
        self.event_handlers["on_all_padding_value_change"] = on_all_padding_value_change

    def _get_css(self) -> str:
        """
        スタイル（CSS）を返す。Vueのstyleに相当。
        必要に応じてoverride可能。
        """
        return ""