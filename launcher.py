#!/usr/bin/env python3
"""
UAV Landing System - Master Launcher

Centralized entry point for all system operations including demos, tools, and utilities.
This script provides easy access to all functionality while keeping the root directory clean.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_script(script_path, args=None):
    """Run a Python script with optional arguments"""
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Script failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"❌ Script not found: {script_path}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="UAV Landing System - Master Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py --main                    # Run main UAV landing system
  python launcher.py --demo end-to-end         # Run end-to-end demo
  python launcher.py --demo complete           # Run complete system demo
  python launcher.py --tool verify             # Verify system integration
  python launcher.py --tool benchmark          # Benchmark GPU performance
  python launcher.py --list                    # List all available operations
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--main", action="store_true", 
                      help="Run main UAV landing system")
    group.add_argument("--demo", choices=["end-to-end", "complete", "summary"],
                      help="Run demonstration scripts")
    group.add_argument("--tool", choices=["verify", "benchmark"],
                      help="Run utility tools")
    group.add_argument("--list", action="store_true",
                      help="List all available operations")
    
    parser.add_argument("--args", nargs="*", 
                       help="Additional arguments to pass to the script")
    
    args = parser.parse_args()
    
    if args.list:
        print_available_operations()
        return True
    
    script_path = None
    description = ""
    
    if args.main:
        script_path = "uav_landing_main.py"
        description = "Main UAV Landing System"
    
    elif args.demo:
        demo_scripts = {
            "end-to-end": ("demos/uav_demo_end_to_end.py", "End-to-End Demo with UDD6 Dataset"),
            "complete": ("demos/demo_complete_system.py", "Complete System Demo with Scallop"),
            "summary": ("demos/demo_summary_generator.py", "Demo Summary Generator")
        }
        script_path, description = demo_scripts[args.demo]
    
    elif args.tool:
        tool_scripts = {
            "verify": ("tools/verify_integration.py", "System Integration Verification"),
            "benchmark": ("tools/benchmark_gpu.py", "GPU Performance Benchmark")
        }
        script_path, description = tool_scripts[args.tool]
    
    if script_path:
        print(f"🚀 Running: {description}")
        print(f"📁 Script: {script_path}")
        if args.args:
            print(f"⚙️  Args: {' '.join(args.args)}")
        print()
        
        success = run_script(script_path, args.args)
        return success
    
    return False

def print_available_operations():
    """Print all available operations"""
    print("🚁 UAV Landing System - Available Operations")
    print("=" * 50)
    print()
    
    print("📋 MAIN SYSTEM")
    print("  --main                     Main UAV landing system (production)")
    print()
    
    print("🎬 DEMONSTRATIONS")
    print("  --demo end-to-end         End-to-end demo with UDD6 dataset")
    print("  --demo complete           Complete system with Scallop integration")
    print("  --demo summary            Generate comprehensive demo reports")
    print()
    
    print("🔧 UTILITY TOOLS")
    print("  --tool verify             Verify system integration")
    print("  --tool benchmark          Benchmark GPU performance")
    print()
    
    print("📁 DIRECTORY STRUCTURE")
    print("  demos/                    Demonstration scripts and results")
    print("  tools/                    Utility tools and verification")
    print("  setup/                    Environment setup scripts")
    print("  src/                      Core system source code")
    print("  tests/                    Test scripts and cases")
    print()
    
    print("💡 TIPS")
    print("  • Use --args to pass additional arguments to scripts")
    print("  • All paths are handled automatically")
    print("  • Check individual README.md files for detailed usage")

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
