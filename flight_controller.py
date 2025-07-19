# flight_controller.py
"""
GPS-Free Flight Controller Interface for UAV Landing
Implements relative positioning commands without GPS dependency.
"""

import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import queue


class FlightMode(Enum):
    """Flight modes for the landing system."""
    MANUAL = "manual"
    SEARCH = "search"
    APPROACH = "approach" 
    PRECISION_LAND = "precision_land"
    EMERGENCY = "emergency"


@dataclass
class RelativeCommand:
    """Relative movement command for GPS-free navigation."""
    forward_velocity: float  # m/s (positive = forward)
    right_velocity: float    # m/s (positive = right)
    down_velocity: float     # m/s (positive = down)
    yaw_rate: float         # rad/s (positive = clockwise)
    duration: float         # seconds
    confidence: float       # 0-1, confidence in the command


@dataclass
class VehicleState:
    """Current state of the UAV."""
    altitude_relative: float  # meters above takeoff
    velocity_ned: Tuple[float, float, float]  # North, East, Down m/s
    attitude_euler: Tuple[float, float, float]  # Roll, pitch, yaw in radians
    timestamp: float
    mode: FlightMode


class MockFlightController:
    """
    Mock flight controller for testing GPS-free landing.
    Simulates UAV responses without actual hardware.
    """
    
    def __init__(self, initial_altitude: float = 10.0):
        self.state = VehicleState(
            altitude_relative=initial_altitude,
            velocity_ned=(0.0, 0.0, 0.0),
            attitude_euler=(0.0, 0.0, 0.0),
            timestamp=time.time(),
            mode=FlightMode.MANUAL
        )
        
        # Simulation parameters
        self.position_ned = np.array([0.0, 0.0, -initial_altitude])  # NED frame
        self.velocity_ned = np.array([0.0, 0.0, 0.0])
        self.max_velocity = 5.0  # m/s
        self.max_acceleration = 2.0  # m/s^2
        
        # Command queue
        self.command_queue = queue.Queue()
        self.current_command = None
        self.command_start_time = 0
        
        # Safety limits
        self.min_altitude = 0.5  # meters
        self.max_altitude = 50.0  # meters
        self.geofence_radius = 100.0  # meters from origin
        
        # Simulation thread
        self.running = False
        self.sim_thread = None
        self.dt = 0.1  # 10Hz simulation
    
    def start_simulation(self):
        """Start the flight simulation thread."""
        if not self.running:
            self.running = True
            self.sim_thread = threading.Thread(target=self._simulation_loop)
            self.sim_thread.daemon = True
            self.sim_thread.start()
    
    def stop_simulation(self):
        """Stop the flight simulation."""
        self.running = False
        if self.sim_thread:
            self.sim_thread.join()
    
    def _simulation_loop(self):
        """Main simulation loop."""
        while self.running:
            try:
                self._update_simulation()
                time.sleep(self.dt)
            except Exception as e:
                print(f"Simulation error: {e}")
                break
    
    def _update_simulation(self):
        """Update the simulated vehicle state."""
        current_time = time.time()
        
        # Process commands
        target_velocity = np.array([0.0, 0.0, 0.0])
        
        if self.current_command is None and not self.command_queue.empty():
            try:
                self.current_command = self.command_queue.get_nowait()
                self.command_start_time = current_time
            except queue.Empty:
                pass
        
        if self.current_command is not None:
            elapsed = current_time - self.command_start_time
            
            if elapsed < self.current_command.duration:
                # Convert body frame velocities to NED frame
                # Simplified - assume no rotation for now
                target_velocity[0] = self.current_command.forward_velocity  # North
                target_velocity[1] = self.current_command.right_velocity    # East
                target_velocity[2] = self.current_command.down_velocity     # Down
            else:
                # Command expired
                self.current_command = None
        
        # Apply velocity limits and acceleration
        velocity_error = target_velocity - self.velocity_ned
        max_accel_step = self.max_acceleration * self.dt
        
        for i in range(3):
            if abs(velocity_error[i]) > max_accel_step:
                velocity_error[i] = np.sign(velocity_error[i]) * max_accel_step
        
        self.velocity_ned += velocity_error
        
        # Apply velocity limits
        speed = np.linalg.norm(self.velocity_ned[:2])  # Horizontal speed
        if speed > self.max_velocity:
            self.velocity_ned[:2] *= self.max_velocity / speed
        
        # Limit vertical velocity
        self.velocity_ned[2] = np.clip(self.velocity_ned[2], -3.0, 3.0)
        
        # Update position
        self.position_ned += self.velocity_ned * self.dt
        
        # Apply safety constraints
        # Altitude limits
        if self.position_ned[2] > -self.min_altitude:
            self.position_ned[2] = -self.min_altitude
            self.velocity_ned[2] = min(0, self.velocity_ned[2])
        
        if self.position_ned[2] < -self.max_altitude:
            self.position_ned[2] = -self.max_altitude
            self.velocity_ned[2] = max(0, self.velocity_ned[2])
        
        # Geofence (simplified circular)
        horizontal_distance = np.linalg.norm(self.position_ned[:2])
        if horizontal_distance > self.geofence_radius:
            # Push back towards center
            direction = self.position_ned[:2] / horizontal_distance
            self.position_ned[:2] = direction * self.geofence_radius
            # Zero out velocity in that direction
            if np.dot(self.velocity_ned[:2], direction) > 0:
                self.velocity_ned[:2] -= direction * np.dot(self.velocity_ned[:2], direction)
        
        # Update state
        self.state.altitude_relative = -self.position_ned[2]
        self.state.velocity_ned = tuple(self.velocity_ned)
        self.state.timestamp = current_time
    
    def send_relative_command(self, command: RelativeCommand) -> bool:
        """
        Send a relative movement command to the flight controller.
        
        Args:
            command: RelativeCommand with desired velocities
            
        Returns:
            True if command was accepted
        """
        # Safety checks
        if abs(command.forward_velocity) > self.max_velocity:
            print(f"Forward velocity {command.forward_velocity} exceeds limit {self.max_velocity}")
            return False
        
        if abs(command.right_velocity) > self.max_velocity:
            print(f"Right velocity {command.right_velocity} exceeds limit {self.max_velocity}")
            return False
        
        if command.duration > 10.0:  # Max command duration
            print(f"Command duration {command.duration} exceeds limit 10.0")
            return False
        
        try:
            self.command_queue.put_nowait(command)
            return True
        except queue.Full:
            print("Command queue is full")
            return False
    
    def get_state(self) -> VehicleState:
        """Get current vehicle state."""
        return self.state
    
    def set_mode(self, mode: FlightMode) -> bool:
        """Set flight mode."""
        self.state.mode = mode
        print(f"Flight mode changed to: {mode.value}")
        return True
    
    def emergency_stop(self):
        """Emergency stop - hover in place."""
        emergency_cmd = RelativeCommand(
            forward_velocity=0.0,
            right_velocity=0.0, 
            down_velocity=0.0,
            yaw_rate=0.0,
            duration=1.0,
            confidence=1.0
        )
        self.send_relative_command(emergency_cmd)
        self.set_mode(FlightMode.EMERGENCY)
    
    def land_now(self):
        """Initiate immediate landing."""
        land_cmd = RelativeCommand(
            forward_velocity=0.0,
            right_velocity=0.0,
            down_velocity=1.0,  # Descend at 1 m/s
            yaw_rate=0.0,
            duration=30.0,  # Long duration for landing
            confidence=1.0
        )
        self.send_relative_command(land_cmd)
        self.set_mode(FlightMode.PRECISION_LAND)


class GPSFreeLandingController:
    """
    Main controller that coordinates visual odometry with flight commands.
    """
    
    def __init__(self, flight_controller: MockFlightController):
        self.fc = flight_controller
        self.landing_state = FlightMode.SEARCH
        
        # Landing parameters
        self.approach_altitude = 5.0  # meters
        self.precision_altitude = 2.0  # meters
        self.landing_threshold = 0.5   # meters
        
        # Control parameters
        self.position_gain = 0.5       # Position controller gain
        self.max_approach_speed = 2.0  # m/s
        self.max_precision_speed = 0.5 # m/s
        
        # Landing zone tracking
        self.target_locked = False
        self.target_loss_count = 0
        self.max_target_loss = 10      # frames
        
    def process_landing_decision(self, decision: Dict, motion_info: Dict, 
                               positioning: 'RelativePositioning') -> bool:
        """
        Process landing decision and send appropriate commands.
        
        Args:
            decision: Decision from symbolic engine
            motion_info: Motion information from visual odometry
            positioning: Relative positioning system
            
        Returns:
            True if landing process should continue
        """
        current_altitude = motion_info['altitude']
        
        if decision['status'] == 'TARGET_ACQUIRED':
            zone = decision['zone']
            
            # Calculate relative movement needed
            landing_vector = positioning.get_landing_vector(
                zone['center'], current_altitude
            )
            
            # Determine landing phase based on altitude and accuracy
            if current_altitude > self.approach_altitude:
                return self._handle_approach_phase(landing_vector, current_altitude)
            elif current_altitude > self.precision_altitude:
                return self._handle_precision_phase(landing_vector, current_altitude) 
            else:
                return self._handle_landing_phase(landing_vector, current_altitude)
        
        else:
            return self._handle_no_target(decision)
    
    def _handle_approach_phase(self, landing_vector: Dict, altitude: float) -> bool:
        """Handle high-altitude approach to landing zone."""
        self.fc.set_mode(FlightMode.APPROACH)
        
        distance = landing_vector['distance_meters']
        confidence = landing_vector['confidence']
        
        if confidence < 0.3:
            # Low confidence - hover and search
            hover_cmd = RelativeCommand(0, 0, 0, 0, 1.0, confidence)
            self.fc.send_relative_command(hover_cmd)
            return True
        
        # Calculate approach velocities
        forward_vel = np.clip(
            landing_vector['forward_meters'] * self.position_gain,
            -self.max_approach_speed, self.max_approach_speed
        )
        
        right_vel = np.clip(
            landing_vector['right_meters'] * self.position_gain,
            -self.max_approach_speed, self.max_approach_speed
        )
        
        # Descend slowly while approaching
        down_vel = 0.5 if distance > 5.0 else 0.2
        
        command = RelativeCommand(
            forward_velocity=forward_vel,
            right_velocity=right_vel,
            down_velocity=down_vel,
            yaw_rate=0.0,
            duration=1.0,
            confidence=confidence
        )
        
        return self.fc.send_relative_command(command)
    
    def _handle_precision_phase(self, landing_vector: Dict, altitude: float) -> bool:
        """Handle precision positioning before final landing."""
        self.fc.set_mode(FlightMode.PRECISION_LAND)
        
        distance = landing_vector['distance_meters']
        confidence = landing_vector['confidence']
        
        if confidence < 0.5:
            # Hold position if confidence is low
            hover_cmd = RelativeCommand(0, 0, 0, 0, 1.0, confidence)
            self.fc.send_relative_command(hover_cmd)
            return True
        
        # Precise movements
        forward_vel = np.clip(
            landing_vector['forward_meters'] * self.position_gain * 0.5,
            -self.max_precision_speed, self.max_precision_speed
        )
        
        right_vel = np.clip(
            landing_vector['right_meters'] * self.position_gain * 0.5,
            -self.max_precision_speed, self.max_precision_speed
        )
        
        # Only descend if well-positioned
        down_vel = 0.2 if distance < 1.0 else 0.0
        
        command = RelativeCommand(
            forward_velocity=forward_vel,
            right_velocity=right_vel,
            down_velocity=down_vel,
            yaw_rate=0.0,
            duration=1.0,
            confidence=confidence
        )
        
        return self.fc.send_relative_command(command)
    
    def _handle_landing_phase(self, landing_vector: Dict, altitude: float) -> bool:
        """Handle final landing phase."""
        distance = landing_vector['distance_meters']
        confidence = landing_vector['confidence']
        
        if distance < self.landing_threshold and confidence > 0.7:
            # Final touchdown
            self.fc.land_now()
            return False  # Landing complete
        
        # Continue precision positioning
        return self._handle_precision_phase(landing_vector, altitude)
    
    def _handle_no_target(self, decision: Dict) -> bool:
        """Handle case when no landing target is available."""
        self.target_loss_count += 1
        
        if self.target_loss_count > self.max_target_loss:
            # Lost target for too long - emergency hover
            print(f"Target lost for {self.target_loss_count} frames: {decision.get('reason', 'Unknown')}")
            self.fc.emergency_stop()
            return False
        
        # Hold position and continue searching
        hover_cmd = RelativeCommand(0, 0, 0, 0, 1.0, 0.1)
        self.fc.send_relative_command(hover_cmd)
        return True
    
    def get_landing_status(self) -> Dict:
        """Get current landing status."""
        state = self.fc.get_state()
        
        return {
            'mode': state.mode.value,
            'altitude': state.altitude_relative,
            'velocity': state.velocity_ned,
            'target_locked': self.target_locked,
            'target_loss_count': self.target_loss_count
        }
