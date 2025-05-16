# gradio/eichi_plus/app.py
import os

import argparse
from dataclasses import dataclass, field
from typing import Callable, Any
from collections.abc import Generator

import gradio as gr

import eichi_plus.shared.eichi_plus as eichi_plus_shared

from locales.i18n_extended import translate
from eichi_utils.ui_styles import get_app_css

from eichi_plus.utils import EventEmitter

from eichi_plus.gradio.eichi_plus.components.base import BaseComponent, BaseComponentInput

from eichi_plus.gradio.eichi_plus.components.normal.root import NormalRootComponent

@dataclass
class AppComponentInput(BaseComponentInput):
    share: bool = False
    server: str = "127.0.0.1"
    port: int = 8002
    inbrowser: bool = False
    lang: str = "ja"
    allowed_paths: list[str] = field(default_factory=list)

    def validate(self) -> bool:
        return True

class AppComponent(BaseComponent):
    def __init__(self, emitter: EventEmitter, inpt: AppComponentInput):
        super().__init__(emitter)

        self.input = inpt

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
            normal_component = NormalRootComponent(emitter=self.emitter)
            with gr.Tabs(elem_id="mode_tabs") as mode_tabs:
                # with gr.TabItem(translate("SDXL")):
                #     gr.Markdown(translate("SDXLモードの説明や設定をここに記述できます。"))
                with gr.TabItem(translate("FramePack")):
                    normal_component()
                with gr.TabItem(translate("FramePackF1")):
                    gr.Markdown(translate("F1モードの説明や設定をここに記述できます。"))
                with gr.TabItem(translate("FramePack 1F推論")):
                    gr.Markdown(translate("1Fモードの説明や設定をここに記述できます。"))
                with gr.TabItem(translate("FramePack ループ")):
                    gr.Markdown(translate("ループモードの説明や設定をここに記述できます。"))
                with gr.TabItem(translate("Paint")):
                    gr.Markdown(translate("Paintモードの説明や設定をここに記述できます。"))
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
        return get_app_css()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument("--server", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--inbrowser", action='store_true')
    parser.add_argument("--lang", type=str, default='ja', help="Language: ja, zh-tw, en")
    args, _ = parser.parse_known_args()

    allowed_paths = [os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './outputs')))]

    app_input: AppComponentInput = AppComponentInput(
        share=args.share,
        server=args.server,
        port=args.port,
        inbrowser=args.inbrowser,
        lang=args.lang,
        allowed_paths=allowed_paths,
    )
    if app_input.validate():
        emitter = EventEmitter()
        app = AppComponent(emitter=emitter, inpt=app_input)
        block = app()
        # 起動コード
        try:
            block.launch(
                server_name=app.input.server,
                server_port=app.input.port,
                share=app.input.share,
                allowed_paths=app.input.allowed_paths,
                inbrowser=app.input.inbrowser,
            )
        except OSError as e:
            if "Cannot find empty port" in str(e):
                print("\n======================================================")
                print(translate("エラー: FramePack-eichiは既に起動しています。"))
                print(translate("同時に複数のインスタンスを実行することはできません。"))
                print(translate("現在実行中のアプリケーションを先に終了してください。"))
                print("======================================================\n")
                input(translate("続行するには何かキーを押してください..."))
            else:
                # その他のOSErrorの場合は元のエラーを表示
                print(translate("\nエラーが発生しました: {e}").format(e=e))
                input(translate("続行するには何かキーを押してください..."))
    else:
        print(f"{AppComponentInput.__class__.__name__} validate error", input)