#!/usr/bin/env python3
"""
Fast DROID Dataset Ingestion Script (Direct GCS Access)

This script handles the ingestion of DROID trajectories using direct Google Cloud Storage access,
which is much faster than TFDS. Uses the raw trajectory data directly from GCS.
"""

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List
import argparse
import os
import subprocess
import random
import re

# Add parent directories to path
current_dir = Path(__file__).parent
robodm_root = current_dir.parent.parent
sys.path.insert(0, str(robodm_root))
sys.path.insert(0, str(current_dir.parent))

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tqdm import tqdm
    import numpy as np
    from robodm import Trajectory
except ImportError as e:
    print(f"❌ Missing required dependency: {e}")
    print("Please install dependencies: pip install -e .")
    sys.exit(1)

# Disable GPU to avoid CUDA warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class FastDROIDIngester:
    """Handles fast DROID dataset ingestion with direct GCS access."""
    
    def __init__(self, output_dir: Path, target_size_gb: float = 2.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target data size (configurable GB in bytes)
        self.target_size_bytes = target_size_gb * 1024 * 1024 * 1024
        
        # Metrics tracking
        self.ingestion_start_time = None
        self.ingestion_end_time = None
        self.total_trajectories = 0
        self.successful_ingestions = 0
        self.failed_ingestions = 0
        self.total_data_size = 0
        self.individual_times = []
        self.episode_sizes = {}  # Track individual episode sizes
        
        # GCS base path for DROID raw data
        self.gs_base_path = "gs://gresearch/robotics/droid_raw/1.0.1"
        
        # Cache for episode listings
        self.episode_cache = {}
        
    def _get_episode_listing(self, max_episodes: int = 1000) -> List[str]:
        """Get list of episode paths from GCS following the DROID structure: LAB/{success|failure}/{DATE}/{TIMESTAMP}/."""
        if self.episode_cache:
            return list(self.episode_cache.keys())[:max_episodes]
        
        print("🔍 Discovering episodes from GCS (following DROID structure)...")
        try:
            # Get lab directories
            result = subprocess.run(
                ["gsutil", "ls", f"{self.gs_base_path}/"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                print(f"❌ Failed to list lab directories: {result.stderr}")
                return []
            
            # Extract lab names (directories ending with /)
            lab_dirs = [line.strip() for line in result.stdout.strip().split('\n') 
                       if line.strip().endswith('/') and not line.strip().endswith('.json') and not line.strip().endswith('.ipynb')]
            
            print(f"🔎 Found {len(lab_dirs)} lab directories: {[lab.split('/')[-2] for lab in lab_dirs]}")
            
            episode_paths = []
            
            # For each lab, get success/failure directories
            for lab_dir in lab_dirs:
                lab_result = subprocess.run(
                    ["gsutil", "ls", lab_dir],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if lab_result.returncode != 0:
                    continue
                
                outcome_dirs = [line.strip() for line in lab_result.stdout.strip().split('\n') 
                               if line.strip().endswith('/')]
                
                # For each outcome (success/failure), get date directories
                for outcome_dir in outcome_dirs:
                    date_result = subprocess.run(
                        ["gsutil", "ls", outcome_dir],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if date_result.returncode != 0:
                        continue
                    
                    date_dirs = [line.strip() for line in date_result.stdout.strip().split('\n') 
                                if line.strip().endswith('/')]
                    
                    # For each date, get timestamp directories (these are the actual episodes)
                    for date_dir in date_dirs:
                        timestamp_result = subprocess.run(
                            ["gsutil", "ls", date_dir],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        if timestamp_result.returncode != 0:
                            continue
                        
                        timestamp_dirs = [line.strip() for line in timestamp_result.stdout.strip().split('\n') 
                                         if line.strip().endswith('/')]
                        
                        # Add all timestamp directories as episodes
                        episode_paths.extend(timestamp_dirs)
                        
                        # Stop if we have enough episodes
                        if len(episode_paths) >= max_episodes:
                            break
                    
                    if len(episode_paths) >= max_episodes:
                        break
                
                if len(episode_paths) >= max_episodes:
                    break
            
            # Extract episode IDs from paths
            episode_ids = []
            for path in episode_paths:
                # Extract the full path as episode ID for uniqueness
                episode_id = path.strip('/').replace(self.gs_base_path.strip('/') + '/', '')
                episode_ids.append(episode_id)
            
            self.episode_cache = {ep: f"{self.gs_base_path}/{ep}" for ep in episode_ids}
            print(f"✅ Found {len(episode_ids)} episodes")
            return episode_ids[:max_episodes]
            
        except subprocess.TimeoutExpired:
            print("❌ Timeout while discovering episodes")
            return []
        except Exception as e:
            print(f"❌ Error discovering episodes: {e}")
            return []
    
    def _download_episode(self, episode_id: str) -> Path:
        """Download only the trajectory.h5 file for a single episode from GCS."""
        temp_dir = Path(tempfile.mkdtemp(prefix=f"droid_episode_"))
        
        # The episode_id contains the full path structure, so we use it directly
        gs_path = f"{self.gs_base_path}/{episode_id}/trajectory.h5"
        
        try:
            # Download only the trajectory.h5 file
            result = subprocess.run(
                ["gsutil", "cp", gs_path, str(temp_dir)],
                capture_output=True,
                timeout=60  # 1 minute timeout
            )
            
            if result.returncode != 0:
                print(f"❌ Failed to download {episode_id}: {result.stderr.decode()}")
                return None
            
            # Check if file was downloaded
            trajectory_file = temp_dir / "trajectory.h5"
            if not trajectory_file.exists():
                print(f"❌ Trajectory file not found for {episode_id}")
                return None
            
            return temp_dir
            
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout downloading {episode_id}")
            return None
        except Exception as e:
            print(f"❌ Error downloading {episode_id}: {e}")
            return None
    
    def _estimate_episode_size(self, episode_path: Path) -> int:
        """Estimate the size of an episode in bytes."""
        try:
            # Check if trajectory.h5 exists
            trajectory_file = episode_path / "trajectory.h5"
            if not trajectory_file.exists():
                return 0
            
            # Get file size
            file_size = trajectory_file.stat().st_size
            
            # Estimate total size (trajectory file + metadata + recordings)
            # Typically recordings are much larger than trajectory data
            estimated_size = file_size * 100  # Conservative estimate
            
            return estimated_size
            
        except Exception as e:
            print(f"⚠️  Error estimating size for {episode_path}: {e}")
            return 0
    
    def _create_tfds_episode_data(self, episode_id: str, gs_path: str) -> Dict:
        """Create TFDS-style episode data for compatibility."""
        return {
            "episode_metadata": {
                "file_path": f"{gs_path}/trajectory.h5"
            },
            "language_instruction": f"Robot trajectory {episode_id}",
            "episode_id": episode_id
        }
    
    def _process_trajectory_file(self, trajectory_file: Path, episode_id: str) -> bool:
        """Process a trajectory.h5 file and save it as RoboDM format."""
        try:
            # Directly read the HDF5 file
            import h5py
            from robodm.utils.flatten import recursively_read_hdf5_group, _flatten
            
            with h5py.File(trajectory_file, "r") as f:
                data_unflattened = recursively_read_hdf5_group(f)
            
            # Flatten the data structure to match VLA format
            data_flattened = _flatten(data_unflattened)
            
            # Create trajectory from the flattened data
            from robodm import Trajectory
            
            # Save as RoboDM format
            output_file = self.output_dir / f"{episode_id.replace('/', '_')}.vla"
            Trajectory.from_dict_of_lists(data_flattened, output_file)
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing trajectory {episode_id}: {e}")
            return False
    
    async def ingest_trajectories(self, max_trajectories: int = None) -> Dict[str, Any]:
        """Ingest DROID trajectories with direct GCS access."""
        print("🚀 Starting fast DROID dataset ingestion...")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"💾 Target data size: {self.target_size_bytes / (1024**3):.1f} GB")
        print()
        
        self.ingestion_start_time = time.time()
        
        try:
            # Get available episodes
            episodes = self._get_episode_listing(max_episodes=10000)
            if not episodes:
                raise ValueError("No episodes found")
            
            print(f"📊 Found {len(episodes)} available episodes")
            
            # Shuffle episodes for random selection
            random.shuffle(episodes)
            
            # Estimate how many episodes we need
            if max_trajectories is None:
                # Start with a conservative estimate
                max_trajectories = min(1000, len(episodes))
            
            print(f"🎯 Processing up to {max_trajectories} episodes")
            
            # Process episodes
            current_size = 0
            
            for i, episode_id in enumerate(tqdm(episodes[:max_trajectories], desc="Processing episodes")):
                start_time = time.time()
                
                try:
                    # Download episode (only trajectory.h5)
                    episode_path = self._download_episode(episode_id)
                    if not episode_path:
                        self.failed_ingestions += 1
                        continue
                    
                    # Estimate episode size
                    episode_size = self._estimate_episode_size(episode_path)
                    self.episode_sizes[episode_id] = episode_size
                    
                    # Process the trajectory file
                    trajectory_file = episode_path / "trajectory.h5"
                    if not self._process_trajectory_file(trajectory_file, episode_id):
                        self.failed_ingestions += 1
                        # Clean up temporary files
                        import shutil
                        shutil.rmtree(episode_path, ignore_errors=True)
                        continue
                    
                    # Update metrics
                    self.successful_ingestions += 1
                    self.total_data_size += episode_size
                    current_size += episode_size
                    
                    # Clean up temporary files
                    import shutil
                    shutil.rmtree(episode_path, ignore_errors=True)
                    
                    # Check if we've reached target size
                    if current_size >= self.target_size_bytes:
                        print(f"\n✅ Reached target size: {current_size / (1024**3):.2f} GB")
                        break
                    
                    # Track timing
                    process_time = time.time() - start_time
                    self.individual_times.append(process_time)
                    
                except Exception as e:
                    print(f"❌ Error processing {episode_id}: {e}")
                    self.failed_ingestions += 1
                    continue
                
                self.total_trajectories += 1
            
            self.ingestion_end_time = time.time()
            
            # Generate results
            results = self._generate_results()
            self._display_results(results)
            
            return results
            
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate ingestion results."""
        total_time = self.ingestion_end_time - self.ingestion_start_time if self.ingestion_end_time else 0
        
        return {
            "success": True,
            "total_time": total_time,
            "total_trajectories": self.total_trajectories,
            "successful_ingestions": self.successful_ingestions,
            "failed_ingestions": self.failed_ingestions,
            "total_data_size_bytes": self.total_data_size,
            "total_data_size_gb": self.total_data_size / (1024**3),
            "target_size_gb": self.target_size_bytes / (1024**3),
            "size_percentage": (self.total_data_size / self.target_size_bytes * 100) if self.target_size_bytes > 0 else 0,
            "average_time_per_trajectory": np.mean(self.individual_times) if self.individual_times else 0,
            "throughput_trajectories_per_second": self.total_trajectories / total_time if total_time > 0 else 0,
            "episode_sizes": self.episode_sizes
        }
    
    def _display_results(self, results: Dict[str, Any]):
        """Display formatted results."""
        print("\n" + "="*80)
        print("📊 FAST DROID INGESTION RESULTS")
        print("="*80)
        
        print(f"⏱️  Total time: {results['total_time']:.2f}s")
        print(f"📈 Trajectories processed: {results['total_trajectories']}")
        print(f"✅ Successful: {results['successful_ingestions']}")
        print(f"❌ Failed: {results['failed_ingestions']}")
        print(f"💾 Data size: {results['total_data_size_gb']:.2f} GB")
        print(f"🎯 Target size: {results['target_size_gb']:.2f} GB")
        print(f"📊 Size percentage: {results['size_percentage']:.1f}%")
        print(f"⚡ Average time per trajectory: {results['average_time_per_trajectory']:.2f}s")
        print(f"🚀 Throughput: {results['throughput_trajectories_per_second']:.2f} trajectories/s")
        
        print(f"\n📁 Output directory: {self.output_dir}")
        print("="*80)

async def main():
    """Main function to run the ingestion."""
    parser = argparse.ArgumentParser(description="Fast DROID dataset ingestion with direct GCS access")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for trajectories"
    )
    parser.add_argument(
        "--target-size-gb",
        type=float,
        default=2.0,
        help="Target data size in GB (default: 2.0)"
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Maximum number of trajectories to process (default: auto-calculate based on target size)"
    )
    
    args = parser.parse_args()
    
    # Create ingester
    ingester = FastDROIDIngester(
        output_dir=args.output_dir,
        target_size_gb=args.target_size_gb
    )
    
    # Run ingestion
    results = await ingester.ingest_trajectories(max_trajectories=args.max_trajectories)
    
    # Save results
    results_file = Path(args.output_dir) / "ingestion_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(main()) 