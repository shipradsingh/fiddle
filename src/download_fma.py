import os
import urllib.request
import zipfile
import logging
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_fma():
    """Download and extract FMA small dataset."""
    url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
    zip_path = "fma_small.zip"
    extract_path = "temp_fma"
    
    try:
        # Create data directories
        Path("data/base_audio").mkdir(parents=True, exist_ok=True)
        
        # Download if not exists
        if not os.path.exists(zip_path):
            logger.info("Downloading FMA small dataset...")
            urllib.request.urlretrieve(url, zip_path)
        
        # Extract
        logger.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        logger.info("Download and extraction complete!")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise
    finally:
        # Cleanup zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)

if __name__ == "__main__":
    download_fma()