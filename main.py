# main.py
"""
Main application loop for UAV Landing Zone Detection System
Integrates neural and symbolic engines for real-time landing zone identification.
"""

import cv2
import numpy as np
import time
import argparse
import os
from pathlib import Path

import config
from neural_engine import NeuralEngine
from symbolic_engine import SymbolicEngine


class UAVLandingAssistant:
    """Main application class that orchestrates the landing detection system."""
    
    def __init__(self, video_source=0, output_dir=None):
        """
        Initialize the UAV Landing Assistant.
        
        Args:
            video_source: Video source (0 for webcam, path for video file)
            output_dir: Directory to save debug outputs (optional)
        """
        self.video_source = video_source
        self.output_dir = Path(output_dir) if output_dir else None
        
        # Initialize engines
        print("Initializing Neural Engine...")
        self.neural_engine = NeuralEngine()
        
        print("Initializing Symbolic Engine...")
        self.symbolic_engine = SymbolicEngine()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.frame_count = 0
        
        # Setup output directory if needed
        if self.output_dir and config.SAVE_DEBUG_FRAMES:
            self.output_dir.mkdir(exist_ok=True, parents=True)
            print(f"Debug frames will be saved to: {self.output_dir}")
    
    def visualize_output(self, frame: np.ndarray, seg_map: np.ndarray, decision: dict) -> np.ndarray:
        """
        Draw the segmentation results and decision on the frame for visualization.
        
        Args:
            frame: Original input frame
            seg_map: Segmentation map from neural network
            decision: Decision from symbolic engine
            
        Returns:
            Annotated frame for display
        """
        # Resize frame to match segmentation resolution if needed
        if frame.shape[:2] != seg_map.shape:
            display_frame = cv2.resize(frame, config.INPUT_RESOLUTION)
        else:
            display_frame = frame.copy()
        
        # Create segmentation overlay
        overlay = np.zeros_like(display_frame)
        
        # Color code the segmentation classes
        class_colors = {
            0: (0, 0, 0),        # background - black
            1: (0, 255, 0),      # safe_flat_surface - green
            2: (255, 255, 0),    # unsafe_uneven_surface - yellow
            3: (255, 165, 0),    # low_obstacle - orange
            4: (0, 0, 255),      # high_obstacle - red
        }
        
        for class_id, color in class_colors.items():
            mask = (seg_map == class_id)
            overlay[mask] = color
        
        # Blend overlay with original frame
        alpha = 0.4
        blended = cv2.addWeighted(display_frame, 1-alpha, overlay, alpha, 0)
        
        # Draw decision results
        if decision['status'] == 'TARGET_ACQUIRED':
            zone = decision['zone']
            center = zone['center']
            
            # Draw landing zone
            cv2.circle(blended, center, 10, config.ZONE_COLOR, -1)
            cv2.circle(blended, center, 20, config.ZONE_COLOR, 3)
            
            # Draw bounding rectangle
            x, y, w, h = zone['bounding_rect']
            cv2.rectangle(blended, (x, y), (x+w, y+h), config.ZONE_COLOR, 2)
            
            # Add text information
            text = f"LANDING ZONE (Score: {zone['score']:.2f})"
            cv2.putText(blended, text, (center[0]-80, center[1]-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.ZONE_COLOR, config.FONT_THICKNESS)
            
            # Add zone details
            details = [
                f"Area: {zone['area']:.0f}px",
                f"Stability: {zone['temporal_stability']}/15",
                f"Aspect Ratio: {zone['aspect_ratio']:.2f}"
            ]
            
            for i, detail in enumerate(details):
                y_pos = center[1] + 10 + (i * 20)
                cv2.putText(blended, detail, (center[0]-60, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.ZONE_COLOR, 1)
        
        else:
            # No valid zone found
            text = f"NO VALID ZONE - {decision.get('reason', 'Unknown')}"
            cv2.putText(blended, text, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, (0, 0, 255), config.FONT_THICKNESS)
        
        # Add performance information
        stats_y = 30
        stats = [
            f"Frame: {decision.get('frame_count', 0)}",
            f"Potential: {decision.get('potential_zones', 0)}",
            f"Valid: {decision.get('valid_zones', 0)}",
            f"Confirmed: {decision.get('confirmed_zones', 0)}",
            f"Obstacles: {decision.get('obstacles', 0)}",
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(blended, stat, (10, stats_y + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add FPS counter
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
            self.current_fps = fps
        
        if hasattr(self, 'current_fps'):
            cv2.putText(blended, f"FPS: {self.current_fps:.1f}", (10, blended.shape[0]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return blended
    
    def save_debug_frame(self, frame: np.ndarray, seg_map: np.ndarray, decision: dict):
        """Save debug information to files."""
        if not (self.output_dir and config.SAVE_DEBUG_FRAMES):
            return
        
        frame_name = f"frame_{self.frame_count:06d}"
        
        # Save original frame
        cv2.imwrite(str(self.output_dir / f"{frame_name}_original.jpg"), frame)
        
        # Save segmentation map (colored)
        seg_colored = np.zeros((seg_map.shape[0], seg_map.shape[1], 3), dtype=np.uint8)
        class_colors = {0: (0, 0, 0), 1: (0, 255, 0), 2: (255, 255, 0), 3: (255, 165, 0), 4: (0, 0, 255)}
        for class_id, color in class_colors.items():
            mask = (seg_map == class_id)
            seg_colored[mask] = color
        cv2.imwrite(str(self.output_dir / f"{frame_name}_segmentation.jpg"), seg_colored)
        
        # Save decision info as text
        with open(self.output_dir / f"{frame_name}_decision.txt", 'w') as f:
            f.write(f"Decision: {decision}\n")
    
    def print_performance_stats(self):
        """Print performance statistics from both engines."""
        neural_stats = self.neural_engine.get_performance_stats()
        symbolic_stats = self.symbolic_engine.get_performance_stats()
        
        print("\n=== Performance Statistics ===")
        print(f"Neural Engine - FPS: {neural_stats['fps']:.1f}, "
              f"Avg Time: {neural_stats['avg_inference_time']*1000:.1f}ms")
        print(f"Symbolic Engine - FPS: {symbolic_stats['fps']:.1f}, "
              f"Avg Time: {symbolic_stats['avg_processing_time']*1000:.1f}ms")
        print(f"Total Frames Processed: {self.frame_count}")
    
    def run(self):
        """Main application loop."""
        print(f"Starting UAV Landing Assistant with video source: {self.video_source}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}")
            return
        
        # Set camera properties if using webcam
        if isinstance(self.video_source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_RESOLUTION[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_RESOLUTION[1])
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Press 'q' to quit, 's' to save current frame, 'r' to reset tracking")
        print("Starting main processing loop...\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream or error reading frame")
                    break
                
                self.frame_count += 1
                self.fps_counter += 1
                
                # Process frame through neural network
                start_total = time.time()
                seg_map = self.neural_engine.process_frame(frame)
                
                # Process through symbolic reasoning
                decision = self.symbolic_engine.run(seg_map)
                total_time = time.time() - start_total
                
                # Visualize results
                display_frame = self.visualize_output(frame, seg_map, decision)
                
                # Save debug information if enabled
                if config.SAVE_DEBUG_FRAMES and self.frame_count % 30 == 0:  # Every 30 frames
                    self.save_debug_frame(frame, seg_map, decision)
                
                # Display results
                cv2.imshow("UAV Landing Assistant", display_frame)
                
                # Print periodic updates
                if config.DEBUG_MODE and self.frame_count % 60 == 0:  # Every 2 seconds at 30fps
                    print(f"Frame {self.frame_count}: {decision['status']} "
                          f"(Total time: {total_time*1000:.1f}ms)")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit requested by user")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = int(time.time())
                    filename = f"landing_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('r'):
                    # Reset tracking
                    self.symbolic_engine.reset_temporal_tracking()
                    print("Temporal tracking reset")
                elif key == ord('p'):
                    # Print current performance stats
                    self.print_performance_stats()
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final performance stats
            self.print_performance_stats()
            
            print("UAV Landing Assistant terminated.")


def main():
    """Entry point for the application."""
    parser = argparse.ArgumentParser(description="UAV Landing Zone Detection System")
    parser.add_argument("--video", "-v", default=0,
                       help="Video source (0 for webcam, path for video file)")
    parser.add_argument("--output", "-o", default=None,
                       help="Output directory for debug files")
    parser.add_argument("--config", "-c", default=None,
                       help="Path to configuration file (optional)")
    
    args = parser.parse_args()
    
    # Convert video argument to int if it's a digit (for webcam)
    try:
        video_source = int(args.video)
    except ValueError:
        video_source = args.video
    
    # Initialize and run the application
    app = UAVLandingAssistant(video_source=video_source, output_dir=args.output)
    app.run()


if __name__ == "__main__":
    main()
