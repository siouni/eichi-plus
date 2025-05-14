# eichi_plus/cli/setup.py
import click
import os
from pathlib import Path
from eichi_plus.node.node_util import NodeUtils

@click.command()
@click.option('--node-work-dir', default='node/work', help='作業ディレクトリ（Node.js と Vite を保存）')
@click.option('--node-version', default='20.17.0', help='Node.js のバージョン')
def setup(node_work_dir: str, node_version: str):
    """Node.js と Vite をセットアップして Vue 3 環境を構築"""
    # プロジェクトディレクトリを eichi_plus/ からの相対パスとして解釈
    node_work_dir = Path(__file__).parent.parent / node_work_dir
    os.makedirs(node_work_dir, exist_ok=True)

    # Node.js セットアップ
    node_setup = NodeUtils(node_work_dir, node_version)
    node_setup.setup_node()

@click.command()
@click.option('--components', default='Section,Prompt', help='生成する Vue コンポーネント名（カンマ区切り）')
@click.option('--output-dir', default='components', help='Vue コンポーネントの出力ディレクトリ')
def create_vue(components: str, output_dir: str):
    """空の Vue 3 単一ファイルコンポーネント（.vue）を生成"""
    # 出力ディレクトリを eichi_plus/ からの相対パスとして解釈
    output_dir = Path(__file__).parent.parent / output_dir
    output_dir.mkdir(exist_ok=True)
    component_names = components.split(',')

    vue_template = """
    <template>
      <div class="{component_name_lower}-container">
        <!-- コンポーネントの内容 -->
      </div>
    </template>

    <script>
    export default {{
      name: '{component_name}',
      props: [],
      emits: [],
      setup() {{
        return {{}};
      }}
    }}
    </script>

    <style scoped>
    .{component_name_lower}-container {{
      /* スタイル */
    }}
    </style>
    """

    for name in component_names:
        name = name.strip()
        if not name:
            continue
        file_path = output_dir / f"{name}.vue"
        if file_path.exists():
            click.echo(f"スキップ: {file_path} は既に存在します。")
            continue
        content = vue_template.format(
            component_name=name,
            component_name_lower=name.lower()
        )
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(f"生成: {file_path}")