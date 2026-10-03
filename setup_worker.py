#!/usr/bin/env python3
"""Provision SadTalker from an explicitly separate Python 3.10 virtualenv.

Run with the WORKER Python, never the ComfyUI Python. No third-party imports.
Model hashes are trust-on-first-observation, NOT upstream-authenticated hashes.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

REPOSITORY = 'https://github.com/OpenTalker/SadTalker.git'
REVISION = 'cd4c0465ae0b54a6f85af57f5c65fec9fe23e7f8'
ROOT = Path(__file__).absolute().parent
# Exact URLs from scripts/download_models.sh at REVISION. Conservative byte floors
# detect truncated/error responses, not authenticity. First observed SHA256 is saved.
RELEASE = 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2-rc/'
FACEX = 'https://github.com/xinntao/facexlib/releases/download/'
MODELS = {
    'checkpoints/mapping_00109-model.pth.tar': (RELEASE + 'mapping_00109-model.pth.tar', 100_000_000),
    'checkpoints/mapping_00229-model.pth.tar': (RELEASE + 'mapping_00229-model.pth.tar', 100_000_000),
    'checkpoints/SadTalker_V0.0.2_256.safetensors': (RELEASE + 'SadTalker_V0.0.2_256.safetensors', 500_000_000),
    'checkpoints/SadTalker_V0.0.2_512.safetensors': (RELEASE + 'SadTalker_V0.0.2_512.safetensors', 500_000_000),
    'gfpgan/weights/alignment_WFLW_4HG.pth': (FACEX + 'v0.1.0/alignment_WFLW_4HG.pth', 100_000_000),
    'gfpgan/weights/detection_Resnet50_Final.pth': (FACEX + 'v0.1.0/detection_Resnet50_Final.pth', 100_000_000),
    'gfpgan/weights/GFPGANv1.4.pth': ('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth', 300_000_000),
    'gfpgan/weights/parsing_parsenet.pth': (FACEX + 'v0.2.2/parsing_parsenet.pth', 50_000_000),
}


def run(command, **kwargs):
    print('+ ' + ' '.join(map(str, command)), flush=True)
    return subprocess.run(command, check=True, **kwargs)


def capture(command, **kwargs):
    return subprocess.check_output(command, text=True, **kwargs).strip()


def canonical(path):
    return os.path.normcase(str(Path(path).resolve()))


def validate_worker(version, prefix, base_prefix, comfy_prefix=None):
    if tuple(version[:2]) != (3, 10):
        raise RuntimeError(f'Run this installer using Python 3.10 inside a virtualenv (found {version[0]}.{version[1]}).')
    if canonical(prefix) == canonical(base_prefix):
        raise RuntimeError('Worker must be a virtual environment, not the base Python.')
    if comfy_prefix and canonical(prefix) == canonical(comfy_prefix):
        raise RuntimeError('Refusing to install into the ComfyUI Python environment.')


def check_environment(comfy_python=None):
    comfy_prefix = None
    if comfy_python:
        # -I prevents Python startup environment variables influencing this identity probe.
        comfy_prefix = capture([str(comfy_python), '-I', '-c', 'import sys; print(sys.prefix)'])
    validate_worker(sys.version_info, sys.prefix, sys.base_prefix, comfy_prefix)
    venv_config = Path(sys.prefix) / 'pyvenv.cfg'
    if not venv_config.is_file() or any(
        line.strip().lower().replace(' ', '') == 'include-system-site-packages=true'
        for line in venv_config.read_text().splitlines()
    ):
        raise RuntimeError('Worker requires a venv without system site packages.')
    if os.environ.get('PYTHONPATH') or os.environ.get('PYTHONHOME'):
        raise RuntimeError('Unset PYTHONPATH and PYTHONHOME before provisioning the worker.')


def _abspath_normcase(path):
    """Normalise path WITHOUT resolving symlinks.

    Two venv python executables that are different symlinks to the same base
    interpreter must compare as distinct (they live in different venvs).
    Resolving would collapse them and allow a second venv to silently overwrite
    a config written by the first.
    """
    return os.path.normcase(os.path.abspath(path))


def check_config(path, python, repository, force):
    if not path.exists() or force:
        return
    try:
        current = json.loads(path.read_text())
        # Compare python by symlink path (venv identity), directory by resolved path.
        same = (_abspath_normcase(current['python']) == _abspath_normcase(python)
                and canonical(current['sadtalker_dir']) == canonical(repository))
    except (ValueError, KeyError, TypeError):
        same = False
    if not same:
        raise RuntimeError('Existing config targets another worker or is invalid. Use --force-config to replace it.')


def check_ffmpeg(directory):
    env = os.environ.copy()
    if directory:
        if not directory.is_dir():
            raise RuntimeError('FFmpeg directory does not exist.')
        env['PATH'] = str(directory) + os.pathsep + env.get('PATH', '')
    for binary in ('ffmpeg', 'ffprobe'):
        executable = shutil.which(binary, path=str(directory) if directory else env.get('PATH'))
        if not executable:
            raise RuntimeError(f'{binary} not found. Supply --ffmpeg-dir containing both binaries.')
        run([executable, '-version'], env=env, stdout=subprocess.DEVNULL)
    return env


def prepare_repository(path):
    if path.exists():
        if not (path / '.git').is_dir():
            raise RuntimeError('Existing SadTalker directory is not a standalone git clone.')
        remote = capture(['git', '-C', str(path), 'remote', 'get-url', 'origin'])
        if remote.rstrip('/').removesuffix('.git') != REPOSITORY.removesuffix('.git'):
            raise RuntimeError('Existing SadTalker origin is not the official HTTPS repository.')
        if capture(['git', '-C', str(path), 'status', '--porcelain', '--untracked-files=all']):
            raise RuntimeError('Existing SadTalker repository is dirty. Preserve or clean it manually.')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        run(['git', 'clone', '--no-checkout', REPOSITORY, str(path)])
    try:
        capture(['git', '-C', str(path), 'cat-file', '-e', REVISION + '^{commit}'])
    except subprocess.CalledProcessError:
        run(['git', '-C', str(path), 'fetch', 'origin', REVISION])
    run(['git', '-C', str(path), 'checkout', '--detach', REVISION])
    if capture(['git', '-C', str(path), 'rev-parse', 'HEAD']) != REVISION:
        raise RuntimeError('Pinned SadTalker revision verification failed.')


def install_packages():
    # No inherited pip configuration or target/prefix variables may redirect installs.
    env = {key: value for key, value in os.environ.items() if not key.startswith('PIP_')}
    env['PIP_CONFIG_FILE'] = os.devnull
    pip = [sys.executable, '-m', 'pip', '--isolated', '--disable-pip-version-check']
    run(pip + ['install', 'setuptools==69.5.1', 'wheel==0.43.0'], env=env)
    run(pip + ['install', 'torch==2.1.2', 'torchvision==0.16.2', 'torchaudio==2.1.2',
               '--index-url', 'https://download.pytorch.org/whl/cu121'], env=env)
    run(pip + ['install', '--no-build-isolation', '-r', str(ROOT / 'requirements-worker.txt')], env=env)
    run(pip + ['check'], env=env)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_model(path, url, minimum, expected_sha256=None, retries=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size >= minimum:
        digest = sha256(path)
        if expected_sha256 is None or digest == expected_sha256:
            return {'url': url, 'size_bytes': path.stat().st_size, 'sha256': digest}
    error = None
    for attempt in range(retries):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.part', delete=False) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(url, timeout=120) as response:
                    shutil.copyfileobj(response, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            size = temporary.stat().st_size
            if size < minimum:
                raise RuntimeError(f'Download too small: {path.name}: {size} < {minimum}')
            digest = sha256(temporary)
            if expected_sha256 is not None and digest != expected_sha256:
                raise RuntimeError(f'SHA256 mismatch against previously observed model: {path.name}')
            os.replace(temporary, path)
            return {'url': url, 'size_bytes': size, 'sha256': digest}
        except (OSError, ValueError, RuntimeError, http.client.HTTPException) as exc:
            error = exc
            if attempt + 1 < retries:
                print(f'Download retry {attempt + 1}/{retries}: {exc}', flush=True)
                time.sleep(2 ** attempt)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError(f'Failed downloading {url}: {error}') from error


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix=path.name + '.', suffix='.part', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_models(repository, evidence):
    manifest_path = evidence / 'model-manifest.json'
    manifest = {'hash_provenance': 'initially observed locally, NOT upstream authenticated',
                'sadtalker_revision': REVISION, 'models': {}}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    for relative, (url, minimum) in MODELS.items():
        previous = manifest['models'].get(relative, {})
        if previous and previous.get('url') != url:
            raise RuntimeError(f'Model manifest URL changed for {relative}')
        print(f'Checking model {relative}', flush=True)
        manifest['models'][relative] = download_model(repository / relative, url, minimum, previous.get('sha256'))
        # Preserve successful observations even if a later model download fails.
        atomic_write(manifest_path, json.dumps(manifest, indent=2) + '\n')


def smoke_check(repository, env):
    run([sys.executable, '-c',
         'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; '
         'x=torch.ones((2,2),device="cuda"); assert (x @ x).sum().item()==8; '
         'print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'], env=env)
    run([sys.executable, 'inference.py', '--help'], cwd=repository, env=env)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comfy-python', type=Path, default=None, help='Actual ComfyUI interpreter (optional)')
    parser.add_argument('--sadtalker-dir', type=Path, default=None, help='Path to SadTalker repo (optional)')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--ffmpeg-dir', type=Path)
    parser.add_argument('--force-config', action='store_true')
    args = parser.parse_args(argv)
    # Preserve the virtualenv executable path rather than resolving its base-Python symlink.
    python = os.path.abspath(sys.executable)
    if args.sadtalker_dir:
        repository = args.sadtalker_dir.resolve()
    elif Path('/workspace/sadtalker-isolated/SadTalker').is_dir():
        repository = Path('/workspace/sadtalker-isolated/SadTalker').resolve()
    else:
        repository = (ROOT / 'SadTalker').resolve()
    config = args.config.absolute()
    ffmpeg = args.ffmpeg_dir.resolve() if args.ffmpeg_dir else None
    check_environment(args.comfy_python)
    check_config(config, python, repository, args.force_config)
    env = check_ffmpeg(ffmpeg)
    prepare_repository(repository)
    install_packages()
    # Keep installer output outside the upstream clone so reruns remain clean.
    evidence = Path(sys.prefix) / 'sadtalker-provisioning'
    download_models(repository, evidence)
    smoke_check(repository, env)
    atomic_write(evidence / 'requirements-freeze.txt', capture([sys.executable, '-m', 'pip', 'freeze', '--all']) + '\n')
    atomic_write(evidence / 'revision.txt', REVISION + '\n')
    check_config(config, python, repository, args.force_config)
    atomic_write(config, json.dumps({'python': python, 'sadtalker_dir': str(repository),
                                    'ffmpeg_dir': str(ffmpeg) if ffmpeg else None,
                                    'timeout_seconds': 1800}, indent=2) + '\n')
    print(f'Worker ready. Config: {config}\nProvisioning evidence: {evidence}', flush=True)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f'Setup failed: {error}', file=sys.stderr)
        raise SystemExit(1)
