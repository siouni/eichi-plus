# gradio/eichi_plus/components/normal/root.py
import os

from dataclasses import dataclass, field
from typing import Callable, Any
from collections.abc import Generator

import gradio as gr

import eichi_plus.shared.eichi_plus as eichi_plus_shared

from locales.i18n_extended import translate

from eichi_plus.utils import EventEmitter

from eichi_plus.gradio.eichi_plus.components.base import BaseComponent

from eichi_plus.gradio.eichi_plus.components.common.section_frame_size import (
    SectionFrameSizeComponent
)
from eichi_plus.gradio.eichi_plus.components.common.all_padding import (
    AllPaddingComponent
)
from eichi_plus.gradio.eichi_plus.components.common.create_video_length import (
    CreateVideoLengthComponent
)

class NormalRootComponent(BaseComponent):
    def __init__(self, emitter: EventEmitter):
        super().__init__(emitter)
    
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

        section_frame_size = SectionFrameSizeComponent(emitter=self.emitter)

        all_padding = AllPaddingComponent(emitter=self.emitter)

        create_video_length = CreateVideoLengthComponent(emitter=self.emitter)
        with block:
            gr.Markdown(translate("通常モードの説明や設定をここに記述できます。"))
            with gr.Row():
                with gr.Column():
                    section_frame_size()
                with gr.Column():
                    all_padding()
                with gr.Column():
                    create_video_length()
        return block

    def _setup_logic(self):
        """
        イベント登録や動的処理をセットアップする。Vueのscriptに相当。
        """
        pass

    def _get_css(self) -> str:
        """
        スタイル（CSS）を返す。Vueのstyleに相当。
        必要に応じてoverride可能。
        """
        return ""