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

SECTION_FRAME_SIZE_MAP:dict[int, str] = {
    1: translate("1秒 (33フレーム)"),
    2: translate("0.5秒 (17フレーム)"),
}
SECTION_FRAME_SIZE_MAP_REV:dict[str, int] = {
    v: k for k, v in SECTION_FRAME_SIZE_MAP.items()
}

@dataclass
class SectionFrameSizeComponentOutput(BaseComponentOutput):
    section_frame_size: int = 1
    def validate(self) -> bool:
        return True

class SectionFrameSizeComponent(BaseComponent):
    def __init__(self, emitter: EventEmitter):
        super().__init__(emitter=emitter)

        self.section_frame_size: gr.Radio = None

        self.output: SectionFrameSizeComponentOutput = SectionFrameSizeComponentOutput()

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
            self.section_frame_size = gr.Radio(
                choices=[SECTION_FRAME_SIZE_MAP[1], SECTION_FRAME_SIZE_MAP[2]],
                value=SECTION_FRAME_SIZE_MAP[1],
                label=translate("セクションフレームサイズ"),
                info=translate("1秒 = 高品質・通常速度 / 0.5秒 = よりなめらかな動き（実験的機能）")
            )
            self.section_frame_size.change(
                fn=self.event_handlers["on_section_frame_size_change"],
                inputs=[self.section_frame_size],
            )
        return block

    def _setup_logic(self):
        """
        イベント登録や動的処理をセットアップする。Vueのscriptに相当。
        """
        def on_section_frame_size_change(section_frame_size_value):
            self.output.section_frame_size = SECTION_FRAME_SIZE_MAP_REV[section_frame_size_value]
        
        self.event_handlers["on_section_frame_size_change"] = on_section_frame_size_change

    def _get_css(self) -> str:
        """
        スタイル（CSS）を返す。Vueのstyleに相当。
        必要に応じてoverride可能。
        """
        return ""