# gradio/eichi_plus/components/common/section_frame_size.py
import os

from dataclasses import dataclass, field
from typing import Callable, Any
from collections.abc import Generator

import gradio as gr

import eichi_plus.shared.eichi_plus as eichi_plus_shared

from locales.i18n_extended import translate
from eichi_utils.video_mode_settings import get_video_modes

from eichi_plus.utils import EventEmitter

from eichi_plus.gradio.eichi_plus.components.base import BaseComponent, BaseComponentOutput

@dataclass
class CreateVideoLengthComponentOutput(BaseComponentOutput):
    create_video_length: int = 0
    def validate(self) -> bool:
        return True

class CreateVideoLengthComponent(BaseComponent):
    def __init__(self, emitter: EventEmitter):
        super().__init__(emitter)

        self.video_modes: list = get_video_modes()
        self.video_modes_rev: dict[str, int] = {
            v: k for k, v in enumerate(self.video_modes)
        }
        self.section_frame_size: gr.Radio = None

        self.output: CreateVideoLengthComponentOutput = CreateVideoLengthComponentOutput()


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
            self.create_video_length = gr.Radio(
                choices=self.video_modes,
                value=self.video_modes[0],
                label=translate("動画長"),
                info=translate("キーフレーム画像のコピー範囲と動画の長さを設定")
            )
            self.create_video_length.change(
                fn=self.event_handlers["on_create_video_length_change"],
                inputs=[self.create_video_length],
            )
        return block

    def _setup_logic(self):
        """
        イベント登録や動的処理をセットアップする。Vueのscriptに相当。
        """
        def on_create_video_length_change(create_video_length_value):
            self.output.create_video_length = self.video_modes_rev[create_video_length_value]
        
        self.event_handlers["on_create_video_length_change"] = on_create_video_length_change

    def _get_css(self) -> str:
        """
        スタイル（CSS）を返す。Vueのstyleに相当。
        必要に応じてoverride可能。
        """
        return ""