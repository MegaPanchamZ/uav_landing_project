#!/usr/bin/env python3
"""
Enhanced UAV Landing Detector with Neurosymbolic Memory
Integration of advanced memory system for robust landing in challenging conditions
"""

import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent))

from uav_landing_detector import UAVLandingDetector, LandingResult
from neurosymbolic_memory import NeuroSymbolicMemory
import numpy as np
import cv2
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field


@dataclass
class MemoryEnhancedResult(LandingResult):
    """Extended result with memory information"""
    # Memory-specific information
    memory_zones: List[Dict] = field(default_factory=list)
    memory_confidence: float = 0.0
    perception_memory_fusion: str = "perception_only"  # perception_only, memory_only, fused
    memory_status: Dict = field(default_factory=dict)
    
    # Recovery information
    recovery_mode: bool = False
    search_pattern: Optional[str] = None


class MemoryEnhancedUAVDetector(UAVLandingDetector):
    """
    UAV Landing Detector enhanced with neurosymbolic memory system.
    
    Extends the base detector with:
    - Spatial memory of landing zones
    - Temporal pattern recognition  
    - Memory-based prediction when visual input is poor
    - Intelligent search and recovery behaviors
    """
    
    def __init__(self, *args, 
                 enable_memory=True,
                 memory_config=None,
                 memory_persistence_file="uav_memory.json",
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        
        # Memory system
        self.enable_memory = enable_memory
        self.memory_persistence_file = Path(memory_persistence_file)
        
        if self.enable_memory:
            # Initialize memory system
            default_config = {
                'memory_horizon': 300.0,
                'spatial_resolution': 0.5, 
                'confidence_decay_rate': 0.98,
                'min_observations': 2
            }
            if memory_config:
                default_config.update(memory_config)
            
            self.memory = NeuroSymbolicMemory(**default_config)
            
            # Load persistent memory if available
            if self.memory_persistence_file.exists():
                self.memory.load_memory(str(self.memory_persistence_file))
        else:
            self.memory = None
        
        # Enhanced state tracking
        self.visual_confidence_history = []
        self.no_target_count = 0
        self.recovery_mode = False
        self.search_pattern_active = False
        
        # Memory fusion parameters
        self.min_visual_confidence = 0.4  # Below this, start using memory
        self.memory_fusion_threshold = 0.6  # Above this visual confidence, use perception only
        
        print(f"🧠 Memory-Enhanced UAV Detector initialized (memory={'enabled' if enable_memory else 'disabled'})")
    
    def process_frame(self, 
                     image: np.ndarray, 
                     altitude: float,
                     current_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     drone_position: Optional[Tuple[float, float]] = None,
                     drone_heading: float = 0.0) -> MemoryEnhancedResult:
        """
        Enhanced frame processing with memory integration
        
        Args:
            image: Input BGR image from camera
            altitude: Current altitude above ground (meters)
            current_velocity: Current velocity [vx, vy, vz] (m/s)
            drone_position: Current drone position in world coordinates (optional)
            drone_heading: Current drone heading in radians (optional)
            
        Returns:
            MemoryEnhancedResult with detection, navigation, and memory information
        """
        
        start_time = time.time()
        
        # Update memory system with current drone state
        if self.memory and drone_position:
            self.memory.update_drone_state(drone_position, altitude, drone_heading)
        
        # Run base detection
        base_result = super().process_frame(image, altitude, current_velocity)
        
        # Convert to memory-enhanced result
        result = self._convert_to_memory_result(base_result)
        
        # Assess visual confidence and decide on memory usage
        visual_confidence = self._assess_visual_confidence(image, base_result)
        self.visual_confidence_history.append(visual_confidence)
        if len(self.visual_confidence_history) > 10:
            self.visual_confidence_history.pop(0)
        
        # Memory integration
        if self.enable_memory:
            result = self._integrate_memory(result, image, altitude, visual_confidence, drone_position)
        
        # Recovery behavior if needed
        result = self._handle_recovery(result, visual_confidence)
        
        # Update memory with observations
        if self.memory and result.status == "TARGET_ACQUIRED":
            self._update_memory_with_observations(result, altitude, image)
        
        return result
    
    def _convert_to_memory_result(self, base_result: LandingResult) -> MemoryEnhancedResult:
        """Convert base result to memory-enhanced result"""
        
        return MemoryEnhancedResult(
            status=base_result.status,
            confidence=base_result.confidence,
            target_pixel=base_result.target_pixel,
            target_world=base_result.target_world,
            distance=base_result.distance,
            bearing=base_result.bearing,
            forward_velocity=base_result.forward_velocity,
            right_velocity=base_result.right_velocity,
            descent_rate=base_result.descent_rate,
            yaw_rate=base_result.yaw_rate,
            processing_time=base_result.processing_time,
            fps=base_result.fps,
            annotated_image=base_result.annotated_image,
            # Memory-specific defaults
            memory_zones=[],
            memory_confidence=0.0,
            perception_memory_fusion="perception_only",
            memory_status={},
            recovery_mode=False,
            search_pattern=None
        )
    
    def _assess_visual_confidence(self, image: np.ndarray, result: LandingResult) -> float:
        """
        Assess the overall confidence in visual perception
        
        Returns a value between 0-1 where:
        - 1.0: Clear, high-contrast scene with obvious features
        - 0.5: Moderate visual information
        - 0.0: Poor visual conditions (uniform grass, etc.)
        """
        
        # Base confidence from detection result
        detection_confidence = result.confidence if result.status == "TARGET_ACQUIRED" else 0.0
        
        # Visual texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Measure local contrast using standard deviation
        contrast_score = np.std(gray) / 128.0  # Normalize to 0-1
        
        # Measure edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Measure color diversity
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        color_diversity = 1.0 - np.max(hist) / np.sum(hist)  # 1 - dominant color ratio
        
        # Combine metrics
        visual_confidence = (
            detection_confidence * 0.4 +      # Detection quality
            contrast_score * 0.3 +            # Image contrast
            edge_density * 10.0 * 0.2 +       # Edge information
            color_diversity * 0.1              # Color variation
        )
        
        return min(1.0, max(0.0, visual_confidence))
    
    def _integrate_memory(self, 
                         result: MemoryEnhancedResult, 
                         image: np.ndarray,
                         altitude: float,
                         visual_confidence: float,
                         drone_position: Optional[Tuple[float, float]]) -> MemoryEnhancedResult:
        """Integrate memory predictions with current perception"""
        
        # Get memory predictions
        memory_zones = self.memory.predict_zones_from_memory(
            min_confidence=0.2,
            max_zones=3
        )
        result.memory_zones = memory_zones
        
        # Calculate memory confidence
        if drone_position:
            result.memory_confidence = self.memory.get_memory_confidence(drone_position)
        
        # Decide on fusion strategy
        if visual_confidence >= self.memory_fusion_threshold:
            # High visual confidence - use perception only
            result.perception_memory_fusion = "perception_only"
            
        elif visual_confidence <= self.min_visual_confidence:
            # Low visual confidence - rely on memory
            result.perception_memory_fusion = "memory_only"
            
            if memory_zones and result.status in ["NO_TARGET", "UNSAFE"]:
                # Use best memory zone as target
                best_memory_zone = max(memory_zones, key=lambda z: z['confidence'])
                
                result.status = "TARGET_ACQUIRED"
                result.confidence = best_memory_zone['confidence']
                result.target_pixel = best_memory_zone.get('center')
                result.target_world = best_memory_zone.get('world_position')
                
                if result.target_world:
                    result.distance = math.sqrt(result.target_world[0]**2 + result.target_world[1]**2)
                    result.bearing = math.atan2(result.target_world[0], result.target_world[1])
                
        else:
            # Medium confidence - fuse perception and memory
            result.perception_memory_fusion = "fused"
            
            if result.status == "TARGET_ACQUIRED" and memory_zones:
                # Enhance current detection with memory
                memory_boost = min(0.3, result.memory_confidence * 0.2)
                result.confidence = min(1.0, result.confidence + memory_boost)
            
            elif result.status == "NO_TARGET" and memory_zones:
                # Use memory as fallback
                best_memory_zone = max(memory_zones, key=lambda z: z['confidence'])
                if best_memory_zone['confidence'] > 0.4:
                    result.status = "TARGET_ACQUIRED"
                    result.confidence = best_memory_zone['confidence'] * 0.7  # Reduce confidence for memory-only
                    result.target_pixel = best_memory_zone.get('center')
                    result.target_world = best_memory_zone.get('world_position')
                    
                    if result.target_world:
                        result.distance = math.sqrt(result.target_world[0]**2 + result.target_world[1]**2)
                        result.bearing = math.atan2(result.target_world[0], result.target_world[1])
        
        # Get memory status
        result.memory_status = self.memory.get_memory_status()
        
        return result
    
    def _handle_recovery(self, result: MemoryEnhancedResult, visual_confidence: float) -> MemoryEnhancedResult:
        """Handle recovery behaviors when target is lost"""
        
        if result.status == "NO_TARGET":
            self.no_target_count += 1
        else:
            self.no_target_count = 0
            self.recovery_mode = False
            self.search_pattern_active = False
        
        # Enter recovery mode if target lost for multiple frames
        if self.no_target_count >= 5:
            self.recovery_mode = True
            result.recovery_mode = True
            
            # Determine recovery strategy
            avg_visual_confidence = np.mean(self.visual_confidence_history) if self.visual_confidence_history else 0.5
            
            if avg_visual_confidence < 0.3:
                # Poor visual conditions - use memory-guided search
                result.search_pattern = "memory_guided"
                
                if self.memory and self.memory.spatial_memory.memory_zones:
                    # Navigate towards most confident memory zone
                    best_zone = max(self.memory.spatial_memory.memory_zones, 
                                  key=lambda z: z.avg_confidence * z.spatial_stability)
                    
                    if best_zone.avg_confidence > 0.3:
                        # Convert to navigation commands
                        rel_pos = best_zone.world_position
                        distance = math.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
                        
                        if distance > 1.0:
                            # Move towards memory zone
                            result.forward_velocity = np.clip(rel_pos[0] * 0.3, -1.0, 1.0)
                            result.right_velocity = np.clip(rel_pos[1] * 0.3, -1.0, 1.0)
                            result.descent_rate = 0.0  # Don't descend while searching
                            
                        result.status = "SEARCHING_MEMORY"
                
            elif avg_visual_confidence < 0.6:
                # Moderate conditions - systematic search
                result.search_pattern = "spiral_search"
                self.search_pattern_active = True
                
                # Implement simple spiral search
                search_time = time.time() % 20.0  # 20-second cycle
                angle = search_time * 0.5  # Slow rotation
                radius = min(2.0, search_time * 0.1)  # Expanding radius
                
                result.forward_velocity = radius * math.cos(angle) * 0.3
                result.right_velocity = radius * math.sin(angle) * 0.3
                result.yaw_rate = 0.1  # Slow yaw for better observation
                result.descent_rate = 0.0
                
                result.status = "SEARCHING_PATTERN"
            
            else:
                # Good visual conditions - hover and observe
                result.search_pattern = "hover_observe"
                result.forward_velocity = 0.0
                result.right_velocity = 0.0
                result.descent_rate = 0.0
                result.yaw_rate = 0.05  # Slow scan
                
                result.status = "SEARCHING_VISUAL"
        
        return result
    
    def _update_memory_with_observations(self, 
                                       result: MemoryEnhancedResult, 
                                       altitude: float,
                                       image: np.ndarray):
        """Update memory system with current observations"""
        
        if not self.memory or result.status != "TARGET_ACQUIRED":
            return
        
        # Create zone dictionary for memory update
        zone = {
            'center': result.target_pixel,
            'area': 1000,  # Default area estimate
            'confidence': result.confidence
        }
        
        # Estimate environment context
        environment_context = self._analyze_environment_context(image)
        
        # Update memory
        if result.target_world:
            self.memory.observe_zones(
                zones=[zone],
                world_positions=[result.target_world],
                confidences=[result.confidence],
                environment_context=environment_context
            )
    
    def _analyze_environment_context(self, image: np.ndarray) -> Dict:
        """Analyze the current environment for context"""
        
        # Simple environment analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Analyze dominant colors
        h, s, v = cv2.split(hsv)
        
        # Check for grass (green hues)
        grass_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        grass_ratio = np.sum(grass_mask > 0) / grass_mask.size
        
        # Check for concrete/pavement (low saturation)
        concrete_mask = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 50, 200]))
        concrete_ratio = np.sum(concrete_mask > 0) / concrete_mask.size
        
        # Determine environment type
        if grass_ratio > 0.6:
            env_type = "grass_field"
        elif concrete_ratio > 0.5:
            env_type = "concrete_surface"
        elif grass_ratio > 0.3 and concrete_ratio > 0.3:
            env_type = "mixed_terrain"
        else:
            env_type = "unknown"
        
        return {
            'environment_type': env_type,
            'grass_ratio': grass_ratio,
            'concrete_ratio': concrete_ratio,
            'lighting_condition': 'normal',  # Could be enhanced with light analysis
            'image_quality': np.std(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) / 128.0
        }
    
    def save_memory(self):
        """Save current memory state for persistence"""
        if self.memory:
            self.memory.save_memory(str(self.memory_persistence_file))
    
    def get_memory_visualization(self, image_size: Tuple[int, int] = (640, 480)) -> Optional[np.ndarray]:
        """Generate visualization of memory state"""
        
        if not self.memory or not self.memory.spatial_memory.confidence_grid is not None:
            return None
        
        # Create visualization image
        vis_img = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        # Draw confidence grid
        grid = self.memory.spatial_memory.confidence_grid
        if grid is not None:
            # Resize grid to image size
            grid_vis = cv2.resize(grid, image_size, interpolation=cv2.INTER_LINEAR)
            
            # Convert to color map
            grid_colored = cv2.applyColorMap((grid_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
            
            # Overlay on visualization
            vis_img = cv2.addWeighted(vis_img, 0.3, grid_colored, 0.7, 0)
        
        # Draw memory zones
        for zone in self.memory.spatial_memory.memory_zones:
            if zone.avg_confidence > 0.2:
                # Convert world position to image coordinates
                pixel_pos = self.memory._world_to_pixel(zone.world_position)
                
                # Draw zone
                color = (0, int(255 * zone.avg_confidence), 0)
                cv2.circle(vis_img, pixel_pos, int(zone.estimated_size * 5), color, 2)
                
                # Draw confidence text
                text = f"{zone.avg_confidence:.2f}"
                cv2.putText(vis_img, text, (pixel_pos[0] + 15, pixel_pos[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw current drone position
        center = (image_size[0] // 2, image_size[1] // 2)
        cv2.circle(vis_img, center, 10, (255, 255, 255), -1)
        cv2.putText(vis_img, "DRONE", (center[0] + 15, center[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_img
    
    def reset_memory(self):
        """Reset memory system"""
        if self.memory:
            self.memory.reset_memory()
        self.visual_confidence_history = []
        self.no_target_count = 0
        self.recovery_mode = False


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the Memory-Enhanced UAV Detector
    """
    
    # Initialize enhanced detector
    detector = MemoryEnhancedUAVDetector(
        model_path="models/bisenetv2_landing.onnx",  
        enable_visualization=True,
        enable_memory=True,
        memory_config={
            'memory_horizon': 300.0,
            'spatial_resolution': 0.5,
            'confidence_decay_rate': 0.98,
            'min_observations': 2
        }
    )
    
    # Test with synthetic data
    print("🧪 Testing memory-enhanced detector...")
    
    # Simulate flight scenario
    altitude = 10.0
    drone_pos = [0.0, 0.0]
    flight_time = 0.0
    
    print("🎮 Controls: 'q' to quit, 'r' to reset, 'm' for memory viz, 's' for stats")
    
    try:
        while True:
            # Generate synthetic test frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Simulate some visual conditions
            if flight_time % 30 < 10:
                # Good visual conditions - add some structure
                cv2.rectangle(frame, (200, 150), (400, 300), (0, 255, 0), -1)  # Landing zone
            elif flight_time % 30 < 20:
                # Poor visual conditions - mostly uniform
                frame[:] = [60, 120, 60]  # Uniform grass color
                # Add some noise
                noise = np.random.randint(-20, 20, frame.shape, dtype=np.int16)
                frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            # else: random conditions
            
            # Update drone position (simulate movement)
            drone_pos[0] += (np.random.random() - 0.5) * 0.1
            drone_pos[1] += (np.random.random() - 0.5) * 0.1
            
            # Process frame with memory
            result = detector.process_frame(
                frame, 
                altitude=altitude,
                drone_position=tuple(drone_pos),
                drone_heading=0.0
            )
            
            # Print interesting results
            if result.perception_memory_fusion != "perception_only":
                print(f"🧠 Memory active: {result.perception_memory_fusion}, "
                      f"Memory zones: {len(result.memory_zones)}, "
                      f"Status: {result.status}")
            
            if result.recovery_mode:
                print(f"🔄 Recovery mode: {result.search_pattern}")
            
            # Show visualization
            if result.annotated_image is not None:
                cv2.imshow("Memory-Enhanced UAV Detector", result.annotated_image)
            else:
                cv2.imshow("Memory-Enhanced UAV Detector", frame)
            
            # Show memory visualization
            memory_viz = detector.get_memory_visualization()
            if memory_viz is not None:
                cv2.imshow("Memory Visualization", memory_viz)
            
            # Handle keys
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                detector.reset_memory()
                altitude = 10.0
                drone_pos = [0.0, 0.0]
                flight_time = 0
            elif key == ord('m'):
                if detector.memory:
                    status = detector.memory.get_memory_status()
                    print(f"📊 Memory Status: {status}")
            elif key == ord('s'):
                if hasattr(result, 'memory_status'):
                    print(f"📊 Memory Status: {result.memory_status}")
            
            flight_time += 1
            
            # Simulate descent
            if result.descent_rate > 0:
                altitude = max(0.5, altitude - 0.1)
            
            time.sleep(0.1)  # ~10 FPS for demo
            
    except KeyboardInterrupt:
        pass
    
    # Save memory for next run
    detector.save_memory()
    
    cv2.destroyAllWindows()
    print("✅ Memory-enhanced demo completed")
