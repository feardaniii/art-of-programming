"""
Model Downloader Utility for Face Recognition Scripts
Auto-downloads required dlib models if they don't exist
"""
import os
import bz2
import urllib.request
import urllib.error
from pathlib import Path
import time
import sys

# Model URLs
MODELS = {
    'shape_predictor_68_face_landmarks.dat': {
        'url': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
        'compressed': True
    },
    'dlib_face_recognition_resnet_model_v1.dat': {
        'url': 'http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2',
        'compressed': True
    },
    'mmod_human_face_detector.dat': {
        'url': 'http://dlib.net/files/mmod_human_face_detector.dat.bz2',
        'compressed': True
    }
}

class DownloadProgressBar:
    """Simple progress bar for downloads"""
    def __init__(self):
        self.last_percent = 0

    def __call__(self, block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            percent = min(100, int(downloaded * 100 / total_size))

            if percent != self.last_percent:
                self.last_percent = percent
                bar_length = 40
                filled = int(bar_length * percent / 100)
                bar = '█' * filled + '░' * (bar_length - filled)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)

                sys.stdout.write(f'\r   [{bar}] {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)')
                sys.stdout.flush()

                if percent == 100:
                    print()  # New line when complete

def download_with_retry(url, destination, max_retries=3):
    """Download a file with retry logic"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"\n   Retry attempt {attempt + 1}/{max_retries}...")
                time.sleep(2)  # Wait before retry

            # Download with progress bar
            urllib.request.urlretrieve(url, destination, DownloadProgressBar())
            return True

        except urllib.error.ContentTooShortError as e:
            print(f"\n   ⚠️  Download incomplete: {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying...")
                # Clean up partial download
                if os.path.exists(destination):
                    os.remove(destination)
            else:
                raise

        except urllib.error.URLError as e:
            print(f"\n   ⚠️  Network error: {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying...")
                if os.path.exists(destination):
                    os.remove(destination)
            else:
                raise

        except Exception as e:
            print(f"\n   ⚠️  Unexpected error: {e}")
            if attempt < max_retries - 1:
                print(f"   Retrying...")
                if os.path.exists(destination):
                    os.remove(destination)
            else:
                raise

    return False

def download_model(model_name, download_dir='.'):
    """
    Download and extract a dlib model if it doesn't exist

    Args:
        model_name: Name of the model file (e.g., 'shape_predictor_68_face_landmarks.dat')
        download_dir: Directory to save the model (default: current directory)

    Returns:
        Path to the model file
    """
    model_path = Path(download_dir) / model_name

    # Check if model already exists
    if model_path.exists():
        print(f"✅ Model already exists: {model_name}")
        return str(model_path)

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model_info = MODELS[model_name]
    url = model_info['url']
    compressed_path = model_path.with_suffix('.dat.bz2')

    print(f"\n📥 Downloading {model_name}...")
    print(f"   URL: {url}")
    print(f"   This may take a few minutes depending on your connection...")

    try:
        # Download compressed file with retry logic
        success = download_with_retry(url, compressed_path, max_retries=3)

        if not success:
            raise Exception("Download failed after multiple retries")

        print(f"✅ Downloaded: {compressed_path.name}")

        # Decompress if needed
        if model_info['compressed']:
            print(f"📦 Decompressing {compressed_path.name}...")
            with bz2.open(compressed_path, 'rb') as f_in:
                with open(model_path, 'wb') as f_out:
                    # Read in chunks to avoid memory issues
                    chunk_size = 1024 * 1024  # 1MB chunks
                    while True:
                        chunk = f_in.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)

            # Remove compressed file
            compressed_path.unlink()
            print(f"✅ Extracted: {model_name}")

        print(f"✅ Model ready: {model_path}")
        return str(model_path)

    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        print(f"\n💡 Manual download instructions:")
        print(f"   1. Download manually from: {url}")
        print(f"   2. Extract the .bz2 file")
        print(f"   3. Place {model_name} in: {download_dir}")

        # Clean up partial downloads
        if compressed_path.exists():
            compressed_path.unlink()
        if model_path.exists():
            model_path.unlink()
        raise

def ensure_models(*model_names, download_dir='.'):
    """
    Ensure multiple models are downloaded

    Args:
        *model_names: Variable number of model names
        download_dir: Directory to save models

    Returns:
        Dictionary mapping model names to their paths
    """
    model_paths = {}

    print("\n🔍 Checking required models...")
    for model_name in model_names:
        model_paths[model_name] = download_model(model_name, download_dir)

    print("\n✅ All required models are ready!")
    return model_paths

if __name__ == "__main__":
    # Test the downloader
    print("Testing model downloader...")
    ensure_models(
        'shape_predictor_68_face_landmarks.dat',
        'dlib_face_recognition_resnet_model_v1.dat'
    )
