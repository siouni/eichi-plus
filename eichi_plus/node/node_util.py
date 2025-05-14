# eichi_plus/node/node_utils.py
import requests
import tarfile
import zipfile
import os
import platform
import subprocess
from pathlib import Path

class NodeUtils:
    def __init__(self, node_work_dir: str, node_version: str = "20.17.0"):
        """Node.js と Vite のユーティリティクラス
        
        Args:
            node_work_dir: 作業ディレクトリ（Node.js バイナリと Vite プロジェクトを保存）
            node_version: Node.js のバージョン
        """
        self.system = platform.system().lower()
        arch = platform.machine().lower()
        if arch == "x86_64":
            self.arch = "x64"
        elif arch == "arm64":
            self.arch = "arm64"
        elif arch == "amd64":
            self.arch = "x64"
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        if self.system == "linux":
            self.ext = "tar.xz"
        elif self.system == "darwin":
            self.ext = "tar.gz"
        elif self.system == "windows":
            self.ext = "zip"
        else:
            raise ValueError(f"Unsupported OS: {self.system}")
        
        self.node_work_dir = Path(node_work_dir)
        self.node_work_dir.mkdir(parents=True, exist_ok=True)
        
        self.node_version = node_version
        
        self.node_dir = self.node_work_dir / "bin"
        self.node_dir.mkdir(parents=True, exist_ok=True)
        self.conponents_dir = self.node_work_dir / "conponents"
        self.conponents_dir.mkdir(parents=True, exist_ok=True)

        self.node_bin = self._get_node_bin_path()
        self.npm_bin = self.node_dir / "node_modules" / "npm" / "bin" / "npm-cli.js"

    def _get_node_url(self) -> str:
        """Node.js のダウンロード URL を取得"""
        if self.system == "linux":
            return f"https://nodejs.org/dist/v{self.node_version}/node-v{self.node_version}-linux-{self.arch}.{self.ext}"
        elif self.system == "darwin":
            return f"https://nodejs.org/dist/v{self.node_version}/node-v{self.node_version}-darwin-{self.arch}.{self.ext}"
        elif self.system == "windows":
            return f"https://nodejs.org/dist/v{self.node_version}/node-v{self.node_version}-win-{self.arch}.{self.ext}"
        raise ValueError(f"Unsupported OS: {self.system}")

    def _get_node_bin_path(self) -> Path:
        """Node.js バイナリのパスを取得"""
        if self.system in ("linux", "darwin"):
            return self.node_dir / f"node-v{self.node_version}-{self.system}-{self.arch}" / "bin" / "node"
        elif self.system == "windows":
            return self.node_dir / f"node-v{self.node_version}-win-{self.arch}" / "node.exe"
        raise ValueError(f"Unsupported OS: {self.system}")

    def setup_node(self):
        """Node.js 環境をセットアップ"""
        if self.node_bin.exists():
            print("Node.js already exists, skipping download.")
        else:
            url = self._get_node_url()
            archive_path = self.node_work_dir / f"node-v{self.node_version}.{self.ext}"
            print(f"Downloading Node.js from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(archive_path, "wb") as f:
                f.write(response.content)

            print(f"Extracting {archive_path}...")
            if url.endswith(".tar.gz") or url.endswith(".tar.xz"):
                with tarfile.open(archive_path, "r:*") as tar:
                    tar.extractall(self.node_dir)
            elif url.endswith(".zip"):
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(self.node_dir)

            archive_path.unlink()
            if platform.system().lower() != "windows":
                self.node_bin.chmod(0o755)

        print("Node Path", self.node_bin.parent)
        os.environ["PATH"] = f"{self.node_bin.parent}{os.pathsep}{os.environ.get('PATH', '')}"
        print(f"Node version: {subprocess.check_output([str(self.node_bin), '-v']).decode().strip()}")

    def setup_vite(self, conponent_name: str):
        """Vite プロジェクトをセットアップ"""
        self.conponents_dir.mkdir(parents=True)
        conponent_dir = self.conponents_dir / conponent_name
        if conponent_dir.exists():
            print("Vite project already exists, skipping setup.")
            return

        os.chdir(self.conponents_dir)
        subprocess.run([
            str(self.node_bin),
            str(self.npm_bin),
            "create",
            "vite@latest",
            conponent_name,
            "--",
            "--template",
            "vue"
        ], check=True)
        os.chdir(self.conponents_dir / conponent_name)
        subprocess.run([str(self.node_bin), str(self.npm_bin), "install"], check=True)
        subprocess.run([str(self.node_bin), str(self.npm_bin), "install", "@vitejs/plugin-vue", "--save-dev"], check=True)
