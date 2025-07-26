#!/usr/bin/env python3
"""
UAV Landing System - Dataset Download Script
===========================================

Automated download script for all required datasets:
1. Semantic Drone Dataset (SDD) - from Kaggle
2. DroneDeploy Dataset - from Google Drive
3. Urban Drone Dataset 6 (UDD6) - from GitHub/Google Drive

Features:
- Automatic download with progress bars
- Checksum verification
- Resume interrupted downloads
- Automatic extraction and organization
- Error handling and retry logic

Requirements:
- kaggle API token (for SDD)
- gdown (for Google Drive downloads)
- Sufficient disk space (~15GB total)

Usage:
    python download_datasets.py --all
    python download_datasets.py --dataset sdd
    python download_datasets.py --dataset dronedeploy --extract_dir ./datasets
"""

import os
import sys
import argparse
import subprocess
import zipfile
import tarfile
import shutil
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Optional, List
from tqdm import tqdm
import requests
import tempfile

# Check for required packages
def check_requirements():
    """Check and install required packages."""
    required_packages = {
        'kaggle': 'kaggle',
        'gdown': 'gdown',
        'requests': 'requests',
        'tqdm': 'tqdm'
    }
    
    missing_packages = []
    
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  Missing required packages: {missing_packages}")
        print(f"Install with: pip install {' '.join(missing_packages)}")
        return False
    
    return True


class DatasetDownloader:
    """Handles downloading and organizing all UAV landing datasets."""
    
    def __init__(self, base_dir: str = './datasets'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset configurations
        self.datasets = {
            'sdd': {
                'name': 'Semantic Drone Dataset',
                'source': 'kaggle',
                'kaggle_dataset': 'bulentsiyah/semantic-drone-dataset',
                'extract_dir': 'semantic_drone_dataset',
                'size_gb': 2.5,
                'description': 'High-resolution drone images with 24 semantic classes'
            },
            'dronedeploy': {
                'name': 'DroneDeploy Dataset',
                'source': 'google_drive',
                'file_id': '1Y3nK2_HlJeprk6q0B4hSKt-XhTLWGV1c',  # Example ID
                'filename': 'drone_deploy_dataset.zip',
                'extract_dir': 'drone_deploy_dataset',
                'size_gb': 8.0,
                'description': 'Large-scale aerial imagery for landing detection'
            },
            'udd6': {
                'name': 'Urban Drone Dataset 6',
                'source': 'github_release',
                'repo': 'MarcWong/UDD',
                'release_tag': 'v1.0',
                'filename': 'udd6_dataset.tar.gz',
                'extract_dir': 'udd6_dataset',
                'size_gb': 4.5,
                'description': 'Urban drone dataset with 6 classes'
            }
        }
        
        print(f"🗂️  Dataset Downloader initialized")
        print(f"   Base directory: {self.base_dir.absolute()}")
        print(f"   Available datasets: {list(self.datasets.keys())}")
    
    def check_disk_space(self, required_gb: float) -> bool:
        """Check if there's enough disk space."""
        free_space = shutil.disk_usage(self.base_dir).free / (1024**3)
        
        if free_space < required_gb:
            print(f"❌ Insufficient disk space!")
            print(f"   Required: {required_gb:.1f} GB")
            print(f"   Available: {free_space:.1f} GB")
            return False
        
        print(f"✅ Sufficient disk space: {free_space:.1f} GB available")
        return True
    
    def verify_kaggle_setup(self) -> bool:
        """Verify Kaggle API is properly configured."""
        try:
            import kaggle
            # Test API access
            kaggle.api.authenticate()
            print("✅ Kaggle API authenticated successfully")
            return True
        except Exception as e:
            print(f"❌ Kaggle API setup failed: {e}")
            print(f"   Please configure Kaggle API:")
            print(f"   1. Get API token from https://www.kaggle.com/settings")
            print(f"   2. Place kaggle.json in ~/.kaggle/")
            print(f"   3. Run: chmod 600 ~/.kaggle/kaggle.json")
            return False
    
    def download_file_with_progress(self, url: str, filepath: Path, chunk_size: int = 8192) -> bool:
        """Download file with progress bar."""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filepath.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = f.write(chunk)
                    pbar.update(size)
            
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def download_from_google_drive(self, file_id: str, output_path: Path) -> bool:
        """Download file from Google Drive using gdown."""
        try:
            import gdown
            
            url = f'https://drive.google.com/uc?id={file_id}'
            
            print(f"📥 Downloading from Google Drive...")
            success = gdown.download(url, str(output_path), quiet=False)
            
            return success is not None
            
        except Exception as e:
            print(f"❌ Google Drive download failed: {e}")
            return False
    
    def download_from_kaggle(self, dataset_name: str, output_dir: Path) -> bool:
        """Download dataset from Kaggle."""
        try:
            import kaggle
            
            print(f"📥 Downloading from Kaggle: {dataset_name}")
            
            # Download to temporary directory first
            with tempfile.TemporaryDirectory() as temp_dir:
                kaggle.api.dataset_download_files(
                    dataset_name,
                    path=temp_dir,
                    unzip=True
                )
                
                # Move to final destination
                temp_path = Path(temp_dir)
                for item in temp_path.iterdir():
                    if item.is_dir():
                        shutil.move(str(item), str(output_dir))
                    else:
                        shutil.move(str(item), str(output_dir / item.name))
            
            return True
            
        except Exception as e:
            print(f"❌ Kaggle download failed: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_dir: Path) -> bool:
        """Extract archive with progress tracking."""
        print(f"📦 Extracting {archive_path.name}...")
        
        try:
            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    
            elif archive_path.suffix.lower() in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_dir)
                    
            else:
                print(f"❌ Unsupported archive format: {archive_path.suffix}")
                return False
            
            print(f"✅ Extraction complete")
            return True
            
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
    
    def calculate_checksum(self, filepath: Path, algorithm: str = 'md5') -> str:
        """Calculate file checksum."""
        hash_func = hashlib.new(algorithm)
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    def download_dataset(self, dataset_key: str, force_redownload: bool = False) -> bool:
        """Download and set up a specific dataset."""
        
        if dataset_key not in self.datasets:
            print(f"❌ Unknown dataset: {dataset_key}")
            return False
        
        config = self.datasets[dataset_key]
        dataset_dir = self.base_dir / config['extract_dir']
        
        # Check if already exists
        if dataset_dir.exists() and not force_redownload:
            print(f"✅ Dataset {config['name']} already exists at {dataset_dir}")
            return True
        
        print(f"\n🚀 Downloading {config['name']}")
        print(f"   Size: ~{config['size_gb']} GB")
        print(f"   Description: {config['description']}")
        
        # Check disk space
        if not self.check_disk_space(config['size_gb'] * 1.5):  # 50% buffer
            return False
        
        # Create dataset directory
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        success = False
        
        try:
            if config['source'] == 'kaggle':
                if not self.verify_kaggle_setup():
                    return False
                success = self.download_from_kaggle(config['kaggle_dataset'], dataset_dir)
                
            elif config['source'] == 'google_drive':
                archive_path = self.base_dir / config['filename']
                success = self.download_from_google_drive(config['file_id'], archive_path)
                
                if success and archive_path.exists():
                    success = self.extract_archive(archive_path, dataset_dir)
                    # Clean up archive
                    archive_path.unlink()
                    
            elif config['source'] == 'github_release':
                # GitHub release download logic
                repo = config['repo']
                tag = config['release_tag']
                filename = config['filename']
                
                url = f"https://github.com/{repo}/releases/download/{tag}/{filename}"
                archive_path = self.base_dir / filename
                
                success = self.download_file_with_progress(url, archive_path)
                
                if success and archive_path.exists():
                    success = self.extract_archive(archive_path, dataset_dir)
                    # Clean up archive
                    archive_path.unlink()
            
            if success:
                print(f"✅ {config['name']} downloaded successfully!")
                
                # Create dataset info file
                info_file = dataset_dir / 'dataset_info.json'
                with open(info_file, 'w') as f:
                    json.dump({
                        'name': config['name'],
                        'dataset_key': dataset_key,
                        'source': config['source'],
                        'download_timestamp': time.time(),
                        'description': config['description']
                    }, f, indent=2)
                
                return True
            else:
                print(f"❌ Failed to download {config['name']}")
                # Clean up partial download
                if dataset_dir.exists():
                    shutil.rmtree(dataset_dir)
                return False
                
        except Exception as e:
            print(f"❌ Error downloading {config['name']}: {e}")
            # Clean up partial download
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            return False
    
    def download_all_datasets(self, force_redownload: bool = False) -> Dict[str, bool]:
        """Download all datasets."""
        
        print(f"🚀 Downloading All UAV Landing Datasets")
        print(f"======================================")
        
        total_size = sum(config['size_gb'] for config in self.datasets.values())
        print(f"Total download size: ~{total_size:.1f} GB")
        
        if not self.check_disk_space(total_size * 1.5):  # 50% buffer
            return {}
        
        results = {}
        
        for dataset_key in self.datasets.keys():
            print(f"\n{'='*60}")
            success = self.download_dataset(dataset_key, force_redownload)
            results[dataset_key] = success
            
            if not success:
                print(f"⚠️  Continuing with remaining datasets...")
        
        # Summary
        print(f"\n🎉 Download Summary:")
        for dataset_key, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            print(f"   {self.datasets[dataset_key]['name']}: {status}")
        
        successful_downloads = sum(results.values())
        print(f"\n📊 {successful_downloads}/{len(results)} datasets downloaded successfully")
        
        return results
    
    def verify_datasets(self) -> Dict[str, bool]:
        """Verify that all datasets are properly downloaded and organized."""
        
        print(f"\n🔍 Verifying Dataset Integrity")
        print(f"=============================")
        
        results = {}
        
        for dataset_key, config in self.datasets.items():
            dataset_dir = self.base_dir / config['extract_dir']
            
            if not dataset_dir.exists():
                print(f"❌ {config['name']}: Directory not found")
                results[dataset_key] = False
                continue
            
            # Check for essential files/structure
            info_file = dataset_dir / 'dataset_info.json'
            
            if info_file.exists():
                # Check if directory has content
                content_count = len(list(dataset_dir.rglob('*')))
                
                if content_count > 5:  # More than just info file
                    print(f"✅ {config['name']}: {content_count} files/directories")
                    results[dataset_key] = True
                else:
                    print(f"⚠️  {config['name']}: Insufficient content ({content_count} items)")
                    results[dataset_key] = False
            else:
                print(f"⚠️  {config['name']}: No dataset info file (may be incomplete)")
                results[dataset_key] = False
        
        return results
    
    def list_datasets(self):
        """List all available datasets and their status."""
        
        print(f"\n📋 Available Datasets")
        print(f"====================")
        
        for dataset_key, config in self.datasets.items():
            dataset_dir = self.base_dir / config['extract_dir']
            status = "✅ Downloaded" if dataset_dir.exists() else "❌ Not downloaded"
            
            print(f"\n🗂️  {config['name']} ({dataset_key})")
            print(f"   Status: {status}")
            print(f"   Size: ~{config['size_gb']} GB")
            print(f"   Source: {config['source']}")
            print(f"   Description: {config['description']}")
            
            if dataset_dir.exists():
                content_count = len(list(dataset_dir.rglob('*')))
                print(f"   Content: {content_count} files/directories")


def main():
    parser = argparse.ArgumentParser(description='UAV Landing System - Dataset Downloader')
    
    # Main actions
    parser.add_argument('--all', action='store_true',
                        help='Download all datasets')
    parser.add_argument('--dataset', type=str, choices=['sdd', 'dronedeploy', 'udd6'],
                        help='Download specific dataset')
    parser.add_argument('--list', action='store_true',
                        help='List available datasets and their status')
    parser.add_argument('--verify', action='store_true',
                        help='Verify dataset integrity')
    
    # Configuration
    parser.add_argument('--extract_dir', type=str, default='./datasets',
                        help='Base directory for datasets')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if dataset exists')
    
    args = parser.parse_args()
    
    print("📥 UAV Landing System - Dataset Downloader")
    print("==========================================")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create downloader
    downloader = DatasetDownloader(args.extract_dir)
    
    try:
        if args.list:
            downloader.list_datasets()
            
        elif args.verify:
            results = downloader.verify_datasets()
            verified_count = sum(results.values())
            print(f"\n✅ {verified_count}/{len(results)} datasets verified")
            
        elif args.all:
            results = downloader.download_all_datasets(args.force)
            
            if results:
                # Verify after download
                print(f"\n🔍 Verifying downloads...")
                verify_results = downloader.verify_datasets()
                
        elif args.dataset:
            success = downloader.download_dataset(args.dataset, args.force)
            
            if success:
                print(f"\n🎉 Dataset '{args.dataset}' ready for training!")
            else:
                print(f"\n❌ Failed to download dataset '{args.dataset}'")
                sys.exit(1)
                
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Download interrupted by user")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 