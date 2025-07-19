#!/usr/bin/env python3
"""
Test video generator for UAV Landing Zone Detection System
Creates synthetic test videos with various scenarios for development and testing.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import random
import math


class TestVideoGenerator:
    """Generate synthetic test videos for UAV landing detection."""
    
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        
    def create_ground_texture(self, terrain_type="mixed"):
        """Create a textured ground background."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if terrain_type == "grass":
            # Green grass texture
            base_color = (34, 139, 34)  # Forest green
            frame[:] = base_color
            # Add noise for texture
            noise = np.random.randint(-20, 20, (self.height, self.width, 3))
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
        elif terrain_type == "dirt":
            # Brown dirt texture
            base_color = (101, 67, 33)  # Brown
            frame[:] = base_color
            noise = np.random.randint(-30, 30, (self.height, self.width, 3))
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
        elif terrain_type == "pavement":
            # Gray pavement
            base_color = (105, 105, 105)  # Dark gray
            frame[:] = base_color
            noise = np.random.randint(-10, 10, (self.height, self.width, 3))
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
        else:  # mixed
            # Create patches of different terrain
            for _ in range(5):
                x = random.randint(0, self.width - 100)
                y = random.randint(0, self.height - 100)
                w = random.randint(80, 200)
                h = random.randint(80, 200)
                
                terrain = random.choice(["grass", "dirt", "pavement"])
                patch = self.create_ground_texture(terrain)
                patch_region = patch[0:h, 0:w]
                
                # Ensure we don't go out of bounds
                end_y = min(y + h, self.height)
                end_x = min(x + w, self.width)
                actual_h = end_y - y
                actual_w = end_x - x
                
                frame[y:end_y, x:end_x] = patch_region[0:actual_h, 0:actual_w]
        
        return frame
    
    def add_obstacles(self, frame, obstacle_count=5):
        """Add various obstacles to the frame."""
        for _ in range(obstacle_count):
            obstacle_type = random.choice(["tree", "rock", "building", "vehicle"])
            
            x = random.randint(50, self.width - 100)
            y = random.randint(50, self.height - 100)
            
            if obstacle_type == "tree":
                # Green circular tree
                radius = random.randint(15, 40)
                cv2.circle(frame, (x, y), radius, (0, 100, 0), -1)
                # Tree trunk
                cv2.rectangle(frame, (x-3, y+radius-5), (x+3, y+radius+10), (101, 67, 33), -1)
                
            elif obstacle_type == "rock":
                # Gray irregular rock
                size = random.randint(10, 30)
                pts = np.array([
                    [x-size, y], [x-size//2, y-size], [x+size//2, y-size],
                    [x+size, y], [x+size//2, y+size], [x-size//2, y+size]
                ], np.int32)
                cv2.fillPoly(frame, [pts], (128, 128, 128))
                
            elif obstacle_type == "building":
                # Rectangular building
                w = random.randint(30, 80)
                h = random.randint(40, 100)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (169, 169, 169), -1)
                # Windows
                for wy in range(y+10, y+h-10, 20):
                    for wx in range(x+10, x+w-10, 15):
                        cv2.rectangle(frame, (wx, wy), (wx+8, wy+8), (0, 191, 255), -1)
                
            elif obstacle_type == "vehicle":
                # Car-like obstacle
                w, h = random.randint(40, 60), random.randint(20, 30)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 139), -1)
                # Wheels
                cv2.circle(frame, (x+10, y+h), 8, (0, 0, 0), -1)
                cv2.circle(frame, (x+w-10, y+h), 8, (0, 0, 0), -1)
        
        return frame
    
    def add_safe_zones(self, frame, zone_count=3):
        """Add clearly defined safe landing zones."""
        zones = []
        
        for i in range(zone_count):
            # Try to find a good location that doesn't overlap too much
            attempts = 0
            while attempts < 20:  # Prevent infinite loops
                x = random.randint(100, self.width - 200)
                y = random.randint(100, self.height - 200)
                size = random.randint(60, 120)
                
                # Check if this overlaps too much with existing zones
                overlaps = False
                for existing_zone in zones:
                    ex, ey, esize = existing_zone
                    dist = math.sqrt((x - ex)**2 + (y - ey)**2)
                    if dist < (size + esize) / 2:
                        overlaps = True
                        break
                
                if not overlaps:
                    # Create a clear, flat zone
                    zone_type = random.choice(["circular", "rectangular"])
                    
                    if zone_type == "circular":
                        cv2.circle(frame, (x, y), size//2, (144, 238, 144), -1)  # Light green
                        # Add some texture but keep it smooth
                        mask = np.zeros((self.height, self.width), dtype=np.uint8)
                        cv2.circle(mask, (x, y), size//2, 255, -1)
                        noise = np.random.randint(-5, 5, (self.height, self.width, 3))
                        textured = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                        frame = np.where(mask[:,:,None] == 255, textured, frame)
                        
                    else:  # rectangular
                        half_size = size // 2
                        cv2.rectangle(frame, (x-half_size, y-half_size), 
                                    (x+half_size, y+half_size), (144, 238, 144), -1)
                    
                    zones.append((x, y, size))
                    break
                
                attempts += 1
        
        return frame, zones
    
    def add_motion_effects(self, frame, frame_num):
        """Add subtle camera motion effects."""
        # Simulate slight camera shake/movement
        shake_x = int(3 * math.sin(frame_num * 0.1))
        shake_y = int(2 * math.cos(frame_num * 0.15))
        
        # Create transformation matrix
        M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
        frame = cv2.warpAffine(frame, M, (self.width, self.height))
        
        return frame
    
    def add_lighting_variation(self, frame, frame_num, total_frames):
        """Add lighting variations to simulate changing conditions."""
        # Simulate gradual lighting change throughout the video
        brightness_factor = 0.8 + 0.4 * (frame_num / total_frames)  # 0.8 to 1.2
        
        # Convert to float, apply brightness, convert back
        frame_float = frame.astype(np.float32) * brightness_factor
        frame = np.clip(frame_float, 0, 255).astype(np.uint8)
        
        return frame
    
    def generate_scenario_video(self, output_path, duration=30, scenario="mixed"):
        """Generate a complete test video with a specific scenario."""
        total_frames = duration * self.fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
        
        print(f"Generating {scenario} scenario video: {output_path}")
        print(f"Duration: {duration}s, Total frames: {total_frames}")
        
        for frame_num in range(total_frames):
            if frame_num % (self.fps * 5) == 0:  # Print every 5 seconds
                print(f"Progress: {frame_num / total_frames * 100:.1f}%")
            
            # Create base terrain
            if scenario == "urban":
                frame = self.create_ground_texture("pavement")
                frame = self.add_obstacles(frame, obstacle_count=8)
                frame, zones = self.add_safe_zones(frame, zone_count=2)
            elif scenario == "rural":
                frame = self.create_ground_texture("grass")
                frame = self.add_obstacles(frame, obstacle_count=3)
                frame, zones = self.add_safe_zones(frame, zone_count=4)
            elif scenario == "challenging":
                frame = self.create_ground_texture("mixed")
                frame = self.add_obstacles(frame, obstacle_count=12)
                frame, zones = self.add_safe_zones(frame, zone_count=1)
            else:  # mixed
                frame = self.create_ground_texture("mixed")
                frame = self.add_obstacles(frame, obstacle_count=6)
                frame, zones = self.add_safe_zones(frame, zone_count=3)
            
            # Add dynamic effects
            frame = self.add_motion_effects(frame, frame_num)
            frame = self.add_lighting_variation(frame, frame_num, total_frames)
            
            # Write frame
            writer.write(frame)
        
        writer.release()
        print(f"Video generation complete: {output_path}")


def main():
    """Main function to generate test videos."""
    parser = argparse.ArgumentParser(description="Generate test videos for UAV landing detection")
    parser.add_argument("--output", "-o", default="test_videos", 
                       help="Output directory for test videos")
    parser.add_argument("--duration", "-d", type=int, default=30,
                       help="Video duration in seconds (default: 30)")
    parser.add_argument("--resolution", "-r", default="640x480",
                       help="Video resolution (default: 640x480)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames per second (default: 30)")
    parser.add_argument("--scenarios", nargs='+', 
                       default=["mixed", "urban", "rural", "challenging"],
                       help="Scenarios to generate (default: all)")
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
    except ValueError:
        print("Invalid resolution format. Use WIDTHxHEIGHT (e.g., 640x480)")
        return
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize generator
    generator = TestVideoGenerator(width, height, args.fps)
    
    # Generate videos for each scenario
    for scenario in args.scenarios:
        output_file = output_dir / f"test_video_{scenario}.mp4"
        generator.generate_scenario_video(output_file, args.duration, scenario)
    
    print(f"\nAll test videos generated in: {output_dir}")
    print("You can now use these videos to test the UAV landing detection system:")
    print("python main.py --video test_videos/test_video_mixed.mp4")


if __name__ == "__main__":
    main()
