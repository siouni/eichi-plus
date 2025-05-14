# eichi-plus.py
import click
from eichi_plus.cli.setup import setup, create_vue

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

cli.add_command(setup)
cli.add_command(create_vue)
cli.add_command(run_gradio)

if __name__ == '__main__':
    cli()