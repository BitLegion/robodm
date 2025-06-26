#!/usr/bin/env python3
"""
DROID Dataset Benchmark Script

This script benchmarks the RoboDM agentic system on pre-ingested DROID trajectories.
It includes progress bars and detailed metrics display.
"""

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, List
import argparse

# Add parent directories to path
current_dir = Path(__file__).parent
robodm_root = current_dir.parent.parent
sys.path.insert(0, str(robodm_root))
sys.path.insert(0, str(current_dir.parent))

try:
    from tqdm import tqdm
    import numpy as np
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install: pip install tqdm numpy")
    sys.exit(1)

from robodm_agentic.core.robodm_interface import RoboDMInterface
from robodm_agentic.core.agent import RoboDMAgent
from robodm_agentic.clients.llm_client import LLMClient
from robodm_agentic.clients.vlm_client import VLMClient


class DROIDBenchmark:
    """Benchmarks RoboDM agentic system on DROID trajectories."""
    
    def __init__(self, trajectory_dir: Path, output_dir: Path = None):
        self.trajectory_dir = Path(trajectory_dir)
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(tempfile.mkdtemp(prefix="droid_benchmark_"))
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics tracking
        self.benchmark_start_time = None
        self.benchmark_end_time = None
        self.query_times = []
        self.batch_times = []
        self.successful_queries = 0
        self.failed_queries = 0
        
        # Resource management
        self.agent = None
        self.robodm_interface = None
        
    async def setup_agent(self, trajectory_paths: List[str]) -> RoboDMAgent:
        """Set up the RoboDM agent with real models."""
        print("🤖 Setting up RoboDM agent with real models...")
        
        # Create RoboDM interface with trajectory directory
        self.robodm_interface = RoboDMInterface(str(self.trajectory_dir))
        
        # Create LLM client
        print("📝 Initializing LLM client...")
        llm_client = None
        try:
            llm_client = LLMClient(
                model="qwen2.5:7b",
                provider="ollama"
            )
            print("✅ LLM client initialized successfully")
        except Exception as e:
            print(f"❌ LLM client initialization failed: {e}")
            print("   This is required for the benchmark to work.")
            raise
        
        # Create VLM client
        print("🖼️  Initializing VLM client...")
        vlm_client = None
        try:
            vlm_client = VLMClient(
                model="llava:7b",
                provider="ollama"
            )
            print("✅ VLM client initialized successfully")
        except Exception as e:
            print(f"⚠️  VLM client initialization failed: {e}")
            print("   Continuing without VLM support...")
            vlm_client = None
        
        # Create agent
        self.agent = RoboDMAgent(
            robodm_interface=self.robodm_interface,
            llm_client=llm_client,
            vlm_client=vlm_client,
            enable_vision=True
        )
        
        # Test setup
        print("🧪 Testing agent setup...")
        test_results = await self.agent.test_setup()
        print("Agent setup test results:")
        for component, status in test_results.items():
            status_str = "✅" if status else "❌"
            print(f"  {status_str} {component}: {'OK' if status else 'Failed'}")
        
        return self.agent
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.agent:
                self.agent.close()
            if self.robodm_interface and hasattr(self.robodm_interface, 'close_all'):
                self.robodm_interface.close_all()
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")
    
    async def run_individual_queries(self, agent: RoboDMAgent) -> List[Dict[str, Any]]:
        """Run individual queries with progress tracking."""
        print("\n🔍 Running individual query benchmark...")
        
        # Define benchmark queries
        queries = [
            "How many trajectories do we have?",
            "Find all successful trajectories",
            "Show me 10 random trajectories",
            "Count trajectories by length",
            "Find trajectories with hidden views",
            "Show me frames where the robot is grasping",
            "Find trajectories with red objects",
            "Analyze robot actions in trajectories",
            "Compare successful vs failed trajectories",
            "Find trajectories longer than 100 timesteps",
            "Show me the beginning and end of trajectories",
            "Find trajectories with specific features",
            "Get all trajectory metadata",
            "Sample 50 trajectories and analyze their frames",
            "Find trajectories with error conditions"
        ]
        
        results = []
        
        for i, query in enumerate(tqdm(queries, desc="Running queries", unit="query")):
            start_time = time.time()
            try:
                # Process query with timeout
                result = await asyncio.wait_for(agent.query(query), timeout=30.0)
                
                query_time = time.time() - start_time
                self.query_times.append(query_time)
                
                results.append({
                    'query': query,
                    'time': query_time,
                    'success': result.success,
                    'result': result.answer,
                    'error': result.error
                })
                
                if result.success:
                    self.successful_queries += 1
                else:
                    self.failed_queries += 1
                
            except asyncio.TimeoutError:
                query_time = time.time() - start_time
                self.query_times.append(query_time)
                
                results.append({
                    'query': query,
                    'time': query_time,
                    'success': False,
                    'error': 'Query timed out after 30 seconds'
                })
                
                self.failed_queries += 1
                
            except Exception as e:
                query_time = time.time() - start_time
                self.query_times.append(query_time)
                
                results.append({
                    'query': query,
                    'time': query_time,
                    'success': False,
                    'error': str(e)
                })
                
                self.failed_queries += 1
        
        return results
    
    async def run_batch_queries(self, agent: RoboDMAgent) -> Dict[str, Any]:
        """Run batch queries with progress tracking."""
        print("\n📦 Running batch query benchmark...")
        
        batch_queries = [
            "How many trajectories do we have?",
            "Find all successful trajectories",
            "Show me 10 random trajectories",
            "Count trajectories by length",
            "Find trajectories with hidden views"
        ]
        
        start_time = time.time()
        
        try:
            # Check if agent supports batch queries
            if not hasattr(agent, 'batch_query'):
                print("⚠️  Agent does not support batch queries, skipping...")
                return {
                    'success': False,
                    'total_time': 0,
                    'queries': len(batch_queries),
                    'error': 'Agent does not support batch queries'
                }
            
            # Process batch
            results = await asyncio.wait_for(agent.batch_query(batch_queries), timeout=60.0)
            
            batch_time = time.time() - start_time
            self.batch_times.append(batch_time)
            
            # Count successful vs failed queries
            successful_count = sum(1 for r in results if r.success)
            failed_count = len(results) - successful_count
            
            return {
                'success': successful_count > 0,
                'total_time': batch_time,
                'queries': len(batch_queries),
                'successful': successful_count,
                'failed': failed_count,
                'throughput': len(batch_queries) / batch_time if batch_time > 0 else 0,
                'results': [{'query': r.query, 'success': r.success, 'answer': r.answer, 'error': r.error} for r in results]
            }
            
        except Exception as e:
            batch_time = time.time() - start_time
            self.batch_times.append(batch_time)
            
            return {
                'success': False,
                'total_time': batch_time,
                'queries': len(batch_queries),
                'error': str(e)
            }
            
        except asyncio.TimeoutError:
            batch_time = time.time() - start_time
            self.batch_times.append(batch_time)
            
            return {
                'success': False,
                'total_time': batch_time,
                'queries': len(batch_queries),
                'error': 'Batch query timed out after 60 seconds'
            }
    
    async def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete benchmark."""
        print("🚀 Starting DROID dataset benchmark...")
        print(f"📁 Trajectory directory: {self.trajectory_dir}")
        print(f"📁 Output directory: {self.output_dir}")
        print()
        
        self.benchmark_start_time = time.time()
        
        try:
            # Get trajectory paths (support both .vla and .hdf5 files)
            trajectory_paths = list(self.trajectory_dir.glob("*.vla")) + list(self.trajectory_dir.glob("*.hdf5"))
            if not trajectory_paths:
                raise ValueError(f"No trajectory files (.vla or .hdf5) found in {self.trajectory_dir}")
            
            print(f"📊 Found {len(trajectory_paths)} trajectories")
            print(f"   File types: {[p.suffix for p in trajectory_paths[:5]]}{'...' if len(trajectory_paths) > 5 else ''}")
            
            # Set up agent
            agent = await self.setup_agent([str(p) for p in trajectory_paths])
            
            # Run individual queries
            individual_results = await self.run_individual_queries(agent)
            
            # Run batch queries
            batch_results = await self.run_batch_queries(agent)
            
            self.benchmark_end_time = time.time()
            
            # Generate and display metrics
            metrics = self._generate_metrics(individual_results, batch_results)
            self._display_metrics(metrics, individual_results)
            
            # Save results
            self._save_results(metrics, individual_results, batch_results)
            
            return metrics
            
        except Exception as e:
            print(f"\n❌ Benchmark failed: {e}")
            raise
        finally:
            # Always cleanup resources
            self.cleanup()
    
    def _generate_metrics(self, individual_results: List[Dict], batch_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive benchmark metrics."""
        total_time = self.benchmark_end_time - self.benchmark_start_time if self.benchmark_end_time else 0
        
        # Handle division by zero cases
        total_queries = len(individual_results)
        success_rate = (self.successful_queries / total_queries * 100) if total_queries > 0 else 0
        
        avg_query_time = np.mean(self.query_times) if self.query_times else 0
        median_query_time = np.median(self.query_times) if self.query_times else 0
        min_query_time = np.min(self.query_times) if self.query_times else 0
        max_query_time = np.max(self.query_times) if self.query_times else 0
        
        overall_throughput = total_queries / total_time if total_time > 0 else 0
        
        return {
            "benchmark_summary": {
                "total_queries": total_queries,
                "successful_queries": self.successful_queries,
                "failed_queries": self.failed_queries,
                "success_rate": success_rate
            },
            "query_performance": {
                "total_query_time": sum(self.query_times),
                "average_query_time": avg_query_time,
                "median_query_time": median_query_time,
                "min_query_time": min_query_time,
                "max_query_time": max_query_time
            },
            "batch_performance": {
                "batch_total_time": sum(self.batch_times) if self.batch_times else 0,
                "batch_throughput": batch_results.get('throughput', 0),
                "batch_success": batch_results.get('success', False)
            },
            "overall_performance": {
                "total_benchmark_time": total_time,
                "queries_per_second": overall_throughput
            }
        }
    
    def _display_metrics(self, metrics: Dict[str, Any], individual_results: List[Dict] = None):
        """Display formatted metrics."""
        print("\n" + "="*80)
        print("📊 DROID BENCHMARK METRICS")
        print("="*80)
        
        # Summary
        summary = metrics["benchmark_summary"]
        print(f"📈 BENCHMARK SUMMARY:")
        print(f"   Total queries: {summary['total_queries']}")
        print(f"   Successful: {summary['successful_queries']}")
        print(f"   Failed: {summary['failed_queries']}")
        print(f"   Success rate: {summary['success_rate']:.1f}%")
        
        # Query Performance
        query_perf = metrics["query_performance"]
        print(f"\n⏱️  QUERY PERFORMANCE:")
        print(f"   Total query time: {query_perf['total_query_time']:.2f}s")
        print(f"   Average query time: {query_perf['average_query_time']:.3f}s")
        print(f"   Median query time: {query_perf['median_query_time']:.3f}s")
        print(f"   Min query time: {query_perf['min_query_time']:.3f}s")
        print(f"   Max query time: {query_perf['max_query_time']:.3f}s")
        
        # Batch Performance
        batch_perf = metrics["batch_performance"]
        print(f"\n📦 BATCH PERFORMANCE:")
        print(f"   Batch total time: {batch_perf['batch_total_time']:.2f}s")
        print(f"   Batch throughput: {batch_perf['batch_throughput']:.2f} queries/second")
        print(f"   Batch success: {'✅' if batch_perf['batch_success'] else '❌'}")
        
        # Overall Performance
        overall = metrics["overall_performance"]
        print(f"\n🚀 OVERALL PERFORMANCE:")
        print(f"   Total benchmark time: {overall['total_benchmark_time']:.2f}s")
        print(f"   Overall throughput: {overall['queries_per_second']:.2f} queries/second")
        
        # Detailed Results
        if individual_results:
            print(f"\n🔍 DETAILED QUERY RESULTS:")
            print("-" * 80)
            for i, result in enumerate(individual_results, 1):
                status = "✅" if result['success'] else "❌"
                print(f"{i:2d}. {status} {result['query']}")
                print(f"    Time: {result['time']:.3f}s")
                if result['success']:
                    # Truncate long results for display
                    answer = result['result']
                    if len(answer) > 100:
                        answer = answer[:97] + "..."
                    print(f"    Result: {answer}")
                else:
                    print(f"    Error: {result['error']}")
                print()
        
        print("="*80)
        print("✅ Benchmark completed successfully!")
        print("="*80)
    
    def _save_results(self, metrics: Dict[str, Any], individual_results: List[Dict], batch_results: Dict):
        """Save benchmark results to files."""
        # Save metrics
        metrics_file = self.output_dir / "benchmark_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save detailed results
        results_file = self.output_dir / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'metrics': metrics,
                'individual_results': individual_results,
                'batch_results': batch_results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to:")
        print(f"   Metrics: {metrics_file}")
        print(f"   Detailed results: {results_file}")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark RoboDM agentic system on DROID trajectories")
    parser.add_argument(
        "--trajectory-dir", 
        type=str, 
        required=True,
        help="Directory containing pre-ingested DROID trajectories (.vla files)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory for benchmark results (default: temporary directory)"
    )
    
    args = parser.parse_args()
    
    # Validate trajectory directory
    trajectory_dir = Path(args.trajectory_dir)
    if not trajectory_dir.exists():
        print(f"❌ Trajectory directory does not exist: {trajectory_dir}")
        return 1
    
    if not list(trajectory_dir.glob("*.vla")) and not list(trajectory_dir.glob("*.hdf5")):
        print(f"❌ No trajectory files (.vla or .hdf5) found in trajectory directory: {trajectory_dir}")
        return 1
    
    # Create benchmark and run
    benchmark = DROIDBenchmark(trajectory_dir, args.output_dir)
    
    try:
        metrics = await benchmark.run_benchmark()
        return 0
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 