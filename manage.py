#!/usr/bin/env python3
"""
UAV Landing Zone Detection System - Project Management Utility
Provides easy commands for testing, running, and managing the project.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"\n{description}...")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def test_system():
    """Run the test suite."""
    return run_command("uv run python test_system.py", "Running system tests")


def generate_videos(duration=30, scenarios=None):
    """Generate test videos."""
    scenarios_str = " ".join(scenarios) if scenarios else "mixed urban rural challenging"
    cmd = f"uv run python generate_test_video.py --duration {duration} --scenarios {scenarios_str}"
    return run_command(cmd, "Generating test videos")


def run_webcam():
    """Run the system with webcam input."""
    return run_command("uv run python main.py --video 0", "Running with webcam")


def run_with_video(video_path):
    """Run the system with a specific video file."""
    return run_command(f"uv run python main.py --video {video_path}", 
                      f"Running with video: {video_path}")


def setup_project():
    """Setup the project environment."""
    print("Setting up UAV Landing Zone Detection System...")
    
    # Check if uv is available
    try:
        subprocess.run("uv --version", shell=True, check=True, 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ UV package manager is available")
    except subprocess.CalledProcessError:
        print("❌ UV package manager not found. Please install UV first.")
        print("Visit: https://docs.astral.sh/uv/getting-started/installation/")
        return False
    
    # Install dependencies
    if not run_command("uv sync", "Installing dependencies"):
        return False
    
    # Run tests
    if not test_system():
        print("⚠️  Tests failed but setup can continue")
    
    # Generate initial test videos
    if not generate_videos(duration=10):
        print("⚠️  Video generation failed but setup can continue")
    
    print("\n🎉 Project setup completed!")
    print("\nNext steps:")
    print("1. Test with webcam: ./manage.py run-webcam")
    print("2. Test with video: ./manage.py run-video test_videos/test_video_mixed.mp4") 
    print("3. Generate more videos: ./manage.py generate-videos --duration 60")
    print("4. Run tests: ./manage.py test")
    
    return True


def show_status():
    """Show project status and available commands."""
    print("UAV Landing Zone Detection System - Status")
    print("=" * 50)
    
    # Check files
    required_files = [
        "config.py", "main.py", "neural_engine.py", 
        "symbolic_engine.py", "test_system.py"
    ]
    
    print("Core Files:")
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING!")
    
    # Check test videos
    test_video_dir = Path("test_videos")
    if test_video_dir.exists():
        videos = list(test_video_dir.glob("*.mp4"))
        print(f"\nTest Videos: {len(videos)} found")
        for video in videos:
            size_mb = video.stat().st_size / (1024 * 1024)
            print(f"  📹 {video.name} ({size_mb:.1f} MB)")
    else:
        print("\nTest Videos: None found")
    
    # Check virtual environment
    venv_path = Path(".venv")
    if venv_path.exists():
        print("\n✅ Virtual environment is set up")
    else:
        print("\n❌ Virtual environment not found")
    
    print(f"\nTo get started: ./manage.py setup")


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="UAV Landing Zone Detection System - Project Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ./manage.py setup                    # Initial project setup
  ./manage.py test                     # Run test suite
  ./manage.py status                   # Show project status
  ./manage.py run-webcam               # Run with webcam
  ./manage.py run-video video.mp4      # Run with specific video
  ./manage.py generate-videos          # Generate test videos
        """)
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    subparsers.add_parser('setup', help='Setup the project environment')
    
    # Status command
    subparsers.add_parser('status', help='Show project status')
    
    # Test command
    subparsers.add_parser('test', help='Run the test suite')
    
    # Generate videos command
    gen_parser = subparsers.add_parser('generate-videos', help='Generate test videos')
    gen_parser.add_argument('--duration', type=int, default=30, help='Video duration in seconds')
    gen_parser.add_argument('--scenarios', nargs='+', 
                           choices=['mixed', 'urban', 'rural', 'challenging'],
                           help='Scenarios to generate')
    
    # Run with webcam
    subparsers.add_parser('run-webcam', help='Run with webcam input')
    
    # Run with video file
    video_parser = subparsers.add_parser('run-video', help='Run with video file')
    video_parser.add_argument('video_path', help='Path to video file')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'setup':
        success = setup_project()
        sys.exit(0 if success else 1)
        
    elif args.command == 'status':
        show_status()
        
    elif args.command == 'test':
        success = test_system()
        sys.exit(0 if success else 1)
        
    elif args.command == 'generate-videos':
        success = generate_videos(args.duration, args.scenarios)
        sys.exit(0 if success else 1)
        
    elif args.command == 'run-webcam':
        success = run_webcam()
        sys.exit(0 if success else 1)
        
    elif args.command == 'run-video':
        success = run_with_video(args.video_path)
        sys.exit(0 if success else 1)
        
    else:
        parser.print_help()
        print("\nRun './manage.py status' to see project status")


if __name__ == "__main__":
    main()
