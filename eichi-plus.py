# eichi-plus.py

import click
import subprocess
from watchgod import watch

@click.group()
def cli():
    """Eichi Plus CLI ツール for Vue.js and Vite"""
    pass

@click.command()
def run_gradio():
    """Gradio インターフェースを起動"""
    from eichi_plus.gradio.eichi import main as gradio_eichi
    # from eichi_plus.worker.eichi import initialize as eichi_initialize
    # eichi_initialize()
    gradio_eichi()

@click.command()
def run_gradio_plus():
    """Gradio インターフェースを起動"""
    from eichi_plus.gradio.eichi_plus.app import main as gradio_eichi_plus
    # from eichi_plus.worker.eichi import initialize as eichi_initialize
    # eichi_initialize()
    gradio_eichi_plus()

@click.command()
def debug():
    """コード変更を監視し、run-gradio-plusを自動再起動"""
    watch_path = "./eichi_plus"

    proc = None

    def start_process():
        return subprocess.Popen([r"venv\\Scripts\\python.exe", "eichi-plus.py", "run-gradio-plus"])

    proc = start_process()

    try:
        def is_valid_file(path, excluded_extensions=None):
            if excluded_extensions is None:
                excluded_extensions = ['.ts', '.vue']
            return not any(path.endswith(ext) for ext in excluded_extensions)
        
        for changes in watch(watch_path):
            changes = [change for change in changes if is_valid_file(change[1])]
            if changes:
                print("変更検知:", changes)
                if proc:
                    proc.terminate()
                    proc.wait()
                proc = start_process()
    except KeyboardInterrupt:
        print("監視を終了します...")
        if proc:
            proc.terminate()
            proc.wait()

cli.add_command(run_gradio)
cli.add_command(run_gradio_plus)
cli.add_command(debug)

if __name__ == '__main__':
    cli()