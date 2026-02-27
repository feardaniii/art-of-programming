"""
AUTONOMOUS DRIVING - Utilities & Setup Guide
=============================================
Helper tools for the autonomous driving detection project.

Sections:
  1. System Diagnostics  — Check GPU, CUDA, TensorFlow, OpenCV
  2. KITTI Downloader     — Download the dataset directly from S3
  3. Video Downloader     — Grab dashcam footage from YouTube via yt-dlp
  4. CUDA Install Guide   — Step-by-step setup for GPU-accelerated training

Requirements: pip install tensorflow opencv-python yt-dlp
Run: python 3_utils.py
"""

import os
import sys
import platform
import subprocess
import zipfile
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "kitti dataset")

# KITTI Object Detection download links (official S3 mirror)
KITTI_URLS = {
    "images": {
        "url":  "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
        "file": "data_object_image_2.zip",
        "size": "~12 GB",
        "desc": "Left color images (training + testing)",
    },
    "labels": {
        "url":  "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
        "file": "data_object_label_2.zip",
        "size": "~5 MB",
        "desc": "Training labels (bounding box annotations)",
    },
}


# ════════════════════════════════════════════════════════════════
# 1. SYSTEM DIAGNOSTICS
# ════════════════════════════════════════════════════════════════

def run_diagnostics():
    """Check everything needed for training: Python, TF, GPU, OpenCV, disk space."""
    print("=" * 70)
    print("  SYSTEM DIAGNOSTICS")
    print("=" * 70)
    print()

    # --- Python ---
    print(f"  Python:     {sys.version.split()[0]}  ({platform.platform()})")

    # --- TensorFlow ---
    try:
        import tensorflow as tf
        print(f"  TensorFlow: {tf.__version__}")
    except ImportError:
        print("  TensorFlow: NOT INSTALLED")
        print("    -> pip install tensorflow")
        tf = None

    # --- GPU / CUDA ---
    if tf is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                print(f"  GPU:        {gpu.name}")
            # CUDA version
            cuda_version = getattr(tf.sysconfig, 'get_build_info', lambda: {})()
            if isinstance(cuda_version, dict):
                cv = cuda_version.get('cuda_version', 'unknown')
                cdnn = cuda_version.get('cudnn_version', 'unknown')
                print(f"  CUDA:       {cv}")
                print(f"  cuDNN:      {cdnn}")
            print()
            print("  GPU STATUS: READY — training will use GPU acceleration")
        else:
            print("  GPU:        NONE DETECTED")
            print()
            print("  GPU STATUS: CPU-ONLY — training will work but be slower")
            print("  See the CUDA Installation Guide below to enable GPU.")
    print()

    # --- OpenCV ---
    try:
        import cv2
        print(f"  OpenCV:     {cv2.__version__}")
    except ImportError:
        print("  OpenCV:     NOT INSTALLED  ->  pip install opencv-python")

    # --- NumPy ---
    try:
        import numpy as np
        print(f"  NumPy:      {np.__version__}")
    except ImportError:
        print("  NumPy:      NOT INSTALLED  ->  pip install numpy")

    # --- Matplotlib ---
    try:
        import matplotlib
        print(f"  Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("  Matplotlib: NOT INSTALLED  ->  pip install matplotlib")

    # --- yt-dlp ---
    try:
        result = subprocess.run(
            ['yt-dlp', '--version'], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"  yt-dlp:     {result.stdout.strip()}")
        else:
            print("  yt-dlp:     NOT WORKING  ->  pip install yt-dlp")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  yt-dlp:     NOT INSTALLED  ->  pip install yt-dlp")

    print()

    # --- KITTI dataset ---
    train_img = os.path.join(DATASET_DIR, "training", "image_2")
    train_lbl = os.path.join(DATASET_DIR, "training", "label_2")
    test_img = os.path.join(DATASET_DIR, "testing", "image_2")

    print("  KITTI DATASET STATUS")
    print("  " + "-" * 40)

    if os.path.isdir(train_img):
        n = len([f for f in os.listdir(train_img) if f.endswith('.png')])
        print(f"  Training images: {n:,} files")
    else:
        print(f"  Training images: NOT FOUND")

    if os.path.isdir(train_lbl):
        n = len([f for f in os.listdir(train_lbl) if f.endswith('.txt')])
        print(f"  Training labels: {n:,} files")
    else:
        print(f"  Training labels: NOT FOUND")

    if os.path.isdir(test_img):
        n = len([f for f in os.listdir(test_img) if f.endswith('.png')])
        print(f"  Testing images:  {n:,} files")
    else:
        print(f"  Testing images:  NOT FOUND")

    print()

    # --- Quick install command ---
    print("  QUICK INSTALL (all dependencies at once):")
    print("  " + "-" * 40)
    print("  pip install tensorflow opencv-python matplotlib numpy yt-dlp")
    print()


# ════════════════════════════════════════════════════════════════
# 2. KITTI DATASET DOWNLOADER
# ════════════════════════════════════════════════════════════════

def _download_progress(count, block_size, total_size):
    """Progress callback for urllib.request.urlretrieve."""
    mb_done = count * block_size / (1024 * 1024)
    if total_size > 0:
        mb_total = total_size / (1024 * 1024)
        pct = min(count * block_size / total_size * 100, 100)
        sys.stdout.write(f"\r  {mb_done:8.1f} / {mb_total:.0f} MB  ({pct:5.1f}%)")
    else:
        sys.stdout.write(f"\r  {mb_done:8.1f} MB downloaded...")
    sys.stdout.flush()


def download_kitti():
    """Download and extract the KITTI Object Detection dataset."""
    print("=" * 70)
    print("  KITTI DATASET DOWNLOADER")
    print("=" * 70)
    print()
    print("  This will download from the official KITTI S3 mirrors:")
    print(f"    Images: {KITTI_URLS['images']['size']}")
    print(f"    Labels: {KITTI_URLS['labels']['size']}")
    print(f"    Destination: {DATASET_DIR}")
    print()

    os.makedirs(DATASET_DIR, exist_ok=True)

    for key, info in KITTI_URLS.items():
        dest = os.path.join(DATASET_DIR, info['file'])

        # Check if already downloaded
        if os.path.exists(dest):
            print(f"  [{key}] Already downloaded: {info['file']}")
            print(f"         Delete it to re-download.")
        else:
            print(f"  [{key}] Downloading {info['desc']}...")
            print(f"         URL: {info['url']}")
            print(f"         Size: {info['size']}")
            print()

            try:
                urllib.request.urlretrieve(info['url'], dest, _download_progress)
                print(f"\n  Downloaded: {info['file']}")
            except Exception as e:
                print(f"\n  DOWNLOAD FAILED: {e}")
                print(f"  Try manually: open the URL in your browser")
                continue

        # Extract
        extract_dir = DATASET_DIR
        # Check if already extracted
        check_dir = os.path.join(extract_dir, "training", "image_2") if key == "images" \
            else os.path.join(extract_dir, "training", "label_2")

        if os.path.isdir(check_dir) and len(os.listdir(check_dir)) > 100:
            print(f"  [{key}] Already extracted.")
        else:
            print(f"  [{key}] Extracting {info['file']}...")
            try:
                with zipfile.ZipFile(dest, 'r') as z:
                    z.extractall(extract_dir)
                print(f"  [{key}] Extraction complete.")
            except zipfile.BadZipFile:
                print(f"  [{key}] ERROR: Corrupt zip file. Delete and re-download.")
            except Exception as e:
                print(f"  [{key}] Extraction error: {e}")

        print()

    print("  Done! Verify with option 1 (System Diagnostics).")
    print()


# ════════════════════════════════════════════════════════════════
# 3. VIDEO DOWNLOADER (yt-dlp)
# ════════════════════════════════════════════════════════════════

def download_video():
    """Download a dashcam video from YouTube using yt-dlp."""
    print("=" * 70)
    print("  DASHCAM VIDEO DOWNLOADER")
    print("=" * 70)
    print()
    print("  Uses yt-dlp to download videos for testing the detector.")
    print("  Good search terms on YouTube:")
    print("    'dashcam driving POV'")
    print("    'dashcam highway traffic'")
    print("    'dashcam city driving 4K'")
    print()

    # Check yt-dlp is installed
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("  yt-dlp is not installed. Install with:")
        print("    pip install yt-dlp")
        print()
        return

    url = input("  Paste YouTube URL (or Enter to skip): ").strip()
    if not url:
        print("  Skipped.")
        return

    # Output filename
    default_name = "dashcam_video.mp4"
    name = input(f"  Output filename (Enter for '{default_name}'): ").strip()
    if not name:
        name = default_name
    if not name.endswith('.mp4'):
        name += '.mp4'

    output_path = os.path.join(SCRIPT_DIR, name)

    print()
    print(f"  Downloading to: {output_path}")
    print("  This may take a minute...")
    print()

    try:
        cmd = [
            'yt-dlp',
            '-f', 'best[ext=mp4]/best',    # Prefer MP4
            '--no-playlist',                 # Single video only
            '-o', output_path,
            url,
        ]
        result = subprocess.run(cmd, timeout=600)  # 10 min timeout

        if result.returncode == 0 and os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\n  Downloaded: {output_path} ({size_mb:.1f} MB)")
            print(f"  Now run 2_inference.py -> option 3 to process it!")
        else:
            print(f"\n  Download may have failed. Check the URL and try again.")
    except subprocess.TimeoutExpired:
        print("\n  Download timed out after 10 minutes.")
    except Exception as e:
        print(f"\n  Error: {e}")
    print()


# ════════════════════════════════════════════════════════════════
# 4. CUDA / GPU INSTALLATION GUIDE
# ════════════════════════════════════════════════════════════════

def print_cuda_guide():
    """Print a step-by-step CUDA installation guide."""
    print("=" * 70)
    print("  CUDA & GPU SETUP GUIDE")
    print("=" * 70)

    guide = """
  WHY GPU?
  ────────
  Training our KITTI detector:
    CPU:  ~4-8 hours (painful)
    GPU:  ~20-40 minutes (reasonable)
    Colab T4 GPU: ~15-25 minutes (free!)

  If you just want to train quickly, use 4_colab_training.ipynb on Google Colab.
  The guide below is for setting up your LOCAL machine.

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 0: CHECK YOUR GPU
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Only NVIDIA GPUs support CUDA (sorry AMD/Intel).
  Check yours:
    Windows: Task Manager -> Performance -> GPU
    Linux:   lspci | grep -i nvidia
    Both:    nvidia-smi  (if drivers installed)

  Minimum: GTX 1060 6GB or newer
  Recommended: RTX 2060 / RTX 3060 / RTX 4060 or better

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 1: INSTALL NVIDIA DRIVERS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Download latest drivers from:
    https://www.nvidia.com/drivers

  After install, verify:
    nvidia-smi
  You should see your GPU name and driver version.

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 2: INSTALL CUDA TOOLKIT
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  VERSION COMPATIBILITY (this is the #1 pitfall!):

    TensorFlow 2.14-2.15  ->  CUDA 11.8  +  cuDNN 8.7
    TensorFlow 2.16-2.18  ->  CUDA 12.3  +  cuDNN 9.1

  Check your TF version first:
    python -c "import tensorflow as tf; print(tf.__version__)"

  Download CUDA Toolkit:
    https://developer.nvidia.com/cuda-toolkit-archive

  WINDOWS install:
    - Run the installer, select "Custom" install
    - Check: CUDA Runtime, cuBLAS, cuDNN, cuFFT
    - Uncheck: Visual Studio integration (unless you need it)
    - Add to PATH: C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3\\bin

  LINUX install (Ubuntu/Debian):
    # For CUDA 12.3:
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-3

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 3: INSTALL cuDNN
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Download cuDNN (requires free NVIDIA account):
    https://developer.nvidia.com/cudnn

  WINDOWS:
    - Extract the zip
    - Copy files into CUDA directories:
        bin\\cudnn*.dll    -> C:\\...\\CUDA\\v12.3\\bin\\
        include\\cudnn*.h  -> C:\\...\\CUDA\\v12.3\\include\\
        lib\\cudnn*.lib    -> C:\\...\\CUDA\\v12.3\\lib\\x64\\

  LINUX:
    sudo apt-get install libcudnn9-cuda-12

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 4: INSTALL TENSORFLOW (GPU-enabled)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Since TF 2.15+, GPU support is included automatically:

    pip install tensorflow

  For older versions or explicit GPU install:

    pip install tensorflow[and-cuda]   # Linux (auto-installs CUDA libs)
    pip install tensorflow             # Windows (uses system CUDA)

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 5: VERIFY
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Run this in Python:

    import tensorflow as tf
    print("TF version:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices('GPU'))
    print("Built with CUDA:", tf.test.is_built_with_cuda())

  You should see your GPU listed. If not:
    - Recheck CUDA/cuDNN version compatibility
    - Make sure CUDA bin directory is in your PATH
    - Restart your terminal/IDE after PATH changes

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TROUBLESHOOTING
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  "Could not load dynamic library 'cudart64_XXX.dll'"
    -> CUDA version mismatch. Check TF↔CUDA compatibility above.

  "Could not load dynamic library 'cudnn64_X.dll'"
    -> cuDNN not installed or wrong version. Re-install cuDNN.

  "GPU is not detected"
    -> Run nvidia-smi to verify drivers work
    -> Restart Python/terminal after installing CUDA
    -> Check PATH includes CUDA bin directory

  GPU shows 0 MB memory used by TF:
    -> TF allocates memory lazily. Run a small computation to verify.

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SKIP ALL THIS: USE GOOGLE COLAB
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Open 4_colab_training.ipynb in Google Colab for free GPU access.
  Runtime -> Change runtime type -> T4 GPU
  No installation needed. Just run the cells.
"""
    print(guide)


# ════════════════════════════════════════════════════════════════
# 5. GPU STRESS TEST
# ════════════════════════════════════════════════════════════════

def gpu_benchmark():
    """Quick benchmark: compare CPU vs GPU speed on a sample computation."""
    print("=" * 70)
    print("  GPU BENCHMARK")
    print("=" * 70)
    print()

    try:
        import tensorflow as tf
    except ImportError:
        print("  TensorFlow not installed. Run: pip install tensorflow")
        return

    import time

    gpus = tf.config.list_physical_devices('GPU')
    print(f"  Detected GPUs: {len(gpus)}")
    for g in gpus:
        print(f"    {g.name}")
    print()

    # Benchmark: matrix multiplication
    size = 3000
    print(f"  Benchmark: {size}x{size} matrix multiplication (10 iterations)")
    print()

    # CPU
    with tf.device('/CPU:0'):
        a = tf.random.normal([size, size])
        b = tf.random.normal([size, size])
        # Warm up
        _ = tf.matmul(a, b)

        start = time.time()
        for _ in range(10):
            _ = tf.matmul(a, b)
        cpu_time = time.time() - start
        print(f"  CPU:  {cpu_time:.2f}s  ({cpu_time / 10 * 1000:.0f} ms per multiply)")

    # GPU
    if gpus:
        with tf.device('/GPU:0'):
            a = tf.random.normal([size, size])
            b = tf.random.normal([size, size])
            _ = tf.matmul(a, b)  # Warm up + transfer

            start = time.time()
            for _ in range(10):
                c = tf.matmul(a, b)
            # Force GPU sync
            _ = c.numpy()
            gpu_time = time.time() - start
            print(f"  GPU:  {gpu_time:.2f}s  ({gpu_time / 10 * 1000:.0f} ms per multiply)")
            print()

            if gpu_time < cpu_time:
                speedup = cpu_time / gpu_time
                print(f"  GPU is {speedup:.1f}x faster than CPU")
            else:
                print(f"  GPU was slower (overhead may dominate for this size)")
    else:
        print("  GPU:  (no GPU detected, skipping)")
    print()

    # Estimate KITTI training time
    print("  ESTIMATED KITTI TRAINING TIME")
    print("  " + "-" * 40)
    if gpus:
        print(f"  With your GPU:  ~20-40 minutes")
    else:
        print(f"  CPU only:       ~4-8 hours")
    print(f"  Google Colab T4: ~15-25 minutes (free)")
    print()


# ════════════════════════════════════════════════════════════════
# MAIN MENU
# ════════════════════════════════════════════════════════════════

def main():
    while True:
        print()
        print("=" * 60)
        print("  AUTONOMOUS DRIVING — Utilities & Setup")
        print("=" * 60)
        print("  1. System Diagnostics (check GPU, packages, dataset)")
        print("  2. Download KITTI Dataset (~12 GB)")
        print("  3. Download Dashcam Video from YouTube")
        print("  4. CUDA / GPU Installation Guide")
        print("  5. GPU Benchmark (CPU vs GPU speed test)")
        print("  6. Exit")
        print()

        choice = input("  Select (1-6): ").strip()

        if choice == '1':
            print()
            run_diagnostics()
        elif choice == '2':
            print()
            confirm = input("  Download ~12 GB of data? (y/n): ").strip().lower()
            if confirm == 'y':
                download_kitti()
            else:
                print("  Cancelled.")
        elif choice == '3':
            print()
            download_video()
        elif choice == '4':
            print()
            print_cuda_guide()
        elif choice == '5':
            print()
            gpu_benchmark()
        elif choice == '6':
            print("\n  Goodbye!")
            break
        else:
            print("  Invalid choice. Enter 1-6.")


if __name__ == '__main__':
    main()
