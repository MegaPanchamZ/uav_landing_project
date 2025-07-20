#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced UAV Landing System with Scallop Integration

This test suite covers all aspects of the neuro-symbolic UAV landing system.
"""

import pytest
import numpy as np
import cv2
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from enhanced_uav_detector import EnhancedUAVDetector, create_enhanced_detector
from scallop_reasoning_engine import ScallopReasoningEngine, ScallopLandingResult
from scallop_mock import MockScallopContext, MockScallopModule

class TestScallopMock:
    """Test the Scallop mock implementation"""
    
    def test_mock_context_creation(self):
        """Test creation of mock Scallop context"""
        ctx = MockScallopContext(provenance="difftopkproofs", k=3)
        assert ctx.provenance == "difftopkproofs"
        assert ctx.k == 3
        assert len(ctx.facts) == 0
    
    def test_mock_fact_insertion(self):
        """Test fact insertion and retrieval"""
        ctx = MockScallopContext()
        
        # Insert facts
        ctx.add_fact("safe_zone", (100, 200, 0.8))
        ctx.add_fact("obstacle", (300, 400))
        
        assert len(ctx.facts) == 2
        assert ("safe_zone", (100, 200, 0.8)) in ctx.facts
        assert ("obstacle", (300, 400)) in ctx.facts
    
    def test_mock_rule_addition(self):
        """Test rule addition to context"""
        ctx = MockScallopContext()
        
        rule = "landing_safe(x, y) :- safe_zone(x, y, c), c > 0.5"
        ctx.add_rule(rule)
        
        assert rule in ctx.rules
    
    def test_mock_query_execution(self):
        """Test query execution"""
        ctx = MockScallopContext()
        
        # Add facts and rules
        ctx.add_fact("safe_zone", (100, 200, 0.8))
        ctx.add_fact("safe_zone", (300, 400, 0.3))
        ctx.add_rule("landing_safe(x, y) :- safe_zone(x, y, c), c > 0.5")
        
        # Run query
        results = ctx.run_query("landing_safe")
        assert len(results) == 1
        assert results[0] == (100, 200)
    
    def test_mock_module_creation(self):
        """Test creation of mock Scallop module"""
        module = MockScallopModule()
        assert module.ctx is not None
        assert hasattr(module, 'forward')
    
    @pytest.fixture
    def sample_segmentation_data(self):
        """Create sample segmentation data for testing"""
        # Create a 256x256 segmentation map
        seg_map = np.zeros((256, 256), dtype=np.uint8)
        
        # Add grass areas (class 0)
        seg_map[50:150, 50:150] = 0
        seg_map[200:250, 200:250] = 0
        
        # Add obstacles (buildings = class 1)
        seg_map[100:120, 100:120] = 1
        
        # Add confidence map
        confidence_map = np.random.uniform(0.6, 0.9, (256, 256)).astype(np.float32)
        
        return seg_map, confidence_map
    
    def test_mock_reasoning_basic(self, sample_segmentation_data):
        """Test basic mock reasoning functionality"""
        seg_map, confidence_map = sample_segmentation_data
        
        # Create mock context
        ctx = MockScallopContext()
        
        # Simulate reasoning process
        # In real implementation, this would be done by the reasoning engine
        height, width = seg_map.shape
        
        # Find safe zones
        for y in range(0, height, 32):
            for x in range(0, width, 32):
                if seg_map[y, x] == 0:  # Grass
                    conf = confidence_map[y, x]
                    ctx.add_fact("safe_zone", (x, y, conf))
        
        # Add rules
        ctx.add_rule("landing_safe(x, y) :- safe_zone(x, y, c), c > 0.7")
        
        # Query
        results = ctx.run_query("landing_safe")
        assert len(results) > 0
        assert all(len(result) == 2 for result in results)  # (x, y) tuples


class TestScallopReasoningEngine:
    """Test the Scallop Reasoning Engine"""
    
    @pytest.fixture
    def reasoning_engine(self):
        """Create a reasoning engine for testing"""
        return ScallopReasoningEngine(context="commercial")
    
    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data for testing"""
        # Create synthetic segmentation output
        seg_output = np.zeros((256, 256), dtype=np.uint8)
        seg_output[64:192, 64:192] = 0  # Large grass area
        seg_output[100:116, 100:116] = 1  # Small building
        
        # Confidence map
        conf_map = np.random.uniform(0.7, 0.9, (256, 256)).astype(np.float32)
        
        # Image shape
        image_shape = (256, 256)
        
        return seg_output, conf_map, image_shape
    
    def test_engine_initialization(self, reasoning_engine):
        """Test reasoning engine initialization"""
        assert reasoning_engine.context == "commercial"
        assert reasoning_engine.scallop_available is not None
        assert hasattr(reasoning_engine, 'reason')
    
    def test_engine_context_switching(self, reasoning_engine):
        """Test context switching"""
        reasoning_engine.set_context("emergency")
        assert reasoning_engine.context == "emergency"
        
        reasoning_engine.set_context("precision")
        assert reasoning_engine.context == "precision"
    
    def test_reasoning_process(self, reasoning_engine, sample_image_data):
        """Test the reasoning process"""
        seg_output, conf_map, image_shape = sample_image_data
        altitude = 10.0
        
        result = reasoning_engine.reason(
            segmentation_output=seg_output,
            confidence_map=conf_map,
            image_shape=image_shape,
            altitude=altitude
        )
        
        assert isinstance(result, ScallopLandingResult)
        assert result.status in ["TARGET_ACQUIRED", "NO_TARGET", "UNSAFE"]
        assert 0.0 <= result.confidence <= 1.0
    
    def test_reasoning_different_contexts(self, sample_image_data):
        """Test reasoning in different contexts"""
        seg_output, conf_map, image_shape = sample_image_data
        altitude = 10.0
        
        contexts = ["commercial", "emergency", "precision", "delivery"]
        
        for context in contexts:
            engine = ScallopReasoningEngine(context=context)
            result = engine.reason(
                segmentation_output=seg_output,
                confidence_map=conf_map,
                image_shape=image_shape,
                altitude=altitude
            )
            
            assert isinstance(result, ScallopLandingResult)
            assert result.context == context
    
    def test_performance_tracking(self, reasoning_engine, sample_image_data):
        """Test performance tracking"""
        seg_output, conf_map, image_shape = sample_image_data
        
        # Run reasoning multiple times
        for _ in range(5):
            reasoning_engine.reason(
                segmentation_output=seg_output,
                confidence_map=conf_map,
                image_shape=image_shape,
                altitude=10.0
            )
        
        stats = reasoning_engine.get_performance_stats()
        assert "total_reasonings" in stats
        assert stats["total_reasonings"] == 5
        assert "average_reasoning_time" in stats


class TestEnhancedUAVDetector:
    """Test the Enhanced UAV Detector"""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        # Create a 512x512 BGR image
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        cv2.rectangle(image, (100, 100), (200, 200), (0, 255, 0), -1)  # Green area
        cv2.rectangle(image, (300, 300), (400, 400), (128, 128, 128), -1)  # Gray building
        
        return image
    
    @pytest.fixture 
    def mock_base_detector(self):
        """Mock the base UAV detector"""
        with patch('enhanced_uav_detector.UAVLandingDetector') as mock:
            mock_instance = Mock()
            mock_instance._run_segmentation.return_value = np.zeros((512, 512), dtype=np.uint8)
            mock_instance._find_landing_zones.return_value = [
                {'center': (256, 256), 'bbox': (200, 200, 112, 112), 'area': 12544}
            ]
            mock_instance._is_zone_safe.return_value = True
            mock_instance._calculate_zone_score.return_value = 0.8
            mock_instance._pixel_to_world.return_value = (0.0, 0.0, 10.0)
            mock_instance._calculate_distance.return_value = 10.0
            mock_instance._calculate_bearing.return_value = 0.0
            mock_instance._generate_commands.return_value = Mock()
            mock_instance._update_tracking.return_value = None
            mock_instance._update_fps.return_value = 30.0
            mock_instance.enable_visualization = False
            mock_instance.safety_margin = 0.5
            mock_instance.get_performance_stats.return_value = {"fps": 30.0}
            mock_instance.get_segmentation_data.return_value = (None, None, None)
            mock.return_value = mock_instance
            return mock
    
    def test_enhanced_detector_initialization(self, mock_base_detector):
        """Test enhanced detector initialization"""
        detector = EnhancedUAVDetector(context="commercial", use_scallop=False)
        
        assert detector.context == "commercial"
        assert detector.use_scallop == False
        assert hasattr(detector, 'process_frame')
    
    def test_enhanced_detector_with_scallop(self, mock_base_detector):
        """Test enhanced detector with Scallop enabled"""
        detector = EnhancedUAVDetector(context="emergency", use_scallop=True)
        
        assert detector.context == "emergency" 
        # Scallop availability depends on whether real Scallop or mock is used
        assert isinstance(detector.use_scallop, bool)
    
    def test_context_switching(self, mock_base_detector):
        """Test context switching in enhanced detector"""
        detector = EnhancedUAVDetector(context="commercial")
        
        detector.set_context("emergency")
        assert detector.context == "emergency"
        
        detector.set_context("precision")
        assert detector.context == "precision"
    
    def test_frame_processing(self, mock_base_detector, sample_image):
        """Test frame processing with enhanced detector"""
        detector = EnhancedUAVDetector(context="commercial", use_scallop=False)
        
        result = detector.process_frame(sample_image, altitude=10.0)
        
        assert hasattr(result, 'status')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'processing_time')
    
    def test_invalid_inputs(self, mock_base_detector):
        """Test handling of invalid inputs"""
        detector = EnhancedUAVDetector(context="commercial")
        
        # Test None image
        result = detector.process_frame(None, altitude=10.0)
        assert result.status == "ERROR"
        assert result.confidence == 0.0
        
        # Test empty image
        empty_image = np.array([])
        result = detector.process_frame(empty_image, altitude=10.0)
        assert result.status == "ERROR"
        
        # Test invalid altitude
        valid_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = detector.process_frame(valid_image, altitude=-5.0)
        assert result.status == "ERROR"
    
    def test_performance_statistics(self, mock_base_detector, sample_image):
        """Test performance statistics collection"""
        detector = EnhancedUAVDetector(context="commercial")
        
        # Process some frames
        for _ in range(3):
            detector.process_frame(sample_image, altitude=10.0)
        
        stats = detector.get_enhanced_performance_stats()
        
        assert "context" in stats
        assert "scallop_available" in stats
        assert "use_scallop" in stats
        assert stats["context"] == "commercial"
    
    def test_reasoning_explanation(self, mock_base_detector):
        """Test reasoning explanation functionality"""
        detector = EnhancedUAVDetector(context="precision")
        
        explanation = detector.get_reasoning_explanation()
        
        assert "reasoning_engine" in explanation
        assert "context" in explanation
        assert "scallop_available" in explanation
        assert explanation["context"] == "precision"
    
    def test_create_enhanced_detector_convenience(self):
        """Test the convenience function"""
        with patch('enhanced_uav_detector.UAVLandingDetector'):
            detector = create_enhanced_detector(
                context="delivery",
                model_path="test_model.onnx"
            )
            
            assert detector.context == "delivery"
            assert detector.use_scallop == True


class TestIntegrationScenarios:
    """Integration tests for complete workflows"""
    
    @pytest.fixture
    def integration_detector(self):
        """Create detector for integration testing"""
        with patch('enhanced_uav_detector.UAVLandingDetector') as mock:
            # Setup comprehensive mock
            mock_instance = Mock()
            mock_instance._run_segmentation.return_value = self._create_test_segmentation()
            mock_instance._find_landing_zones.return_value = self._create_test_zones()
            mock_instance._is_zone_safe.return_value = True
            mock_instance._calculate_zone_score.return_value = 0.75
            mock_instance._pixel_to_world.return_value = (0.0, 0.0, 10.0)
            mock_instance._calculate_distance.return_value = 10.0
            mock_instance._calculate_bearing.return_value = 45.0
            mock_instance._generate_commands.return_value = Mock(
                status="TARGET_ACQUIRED",
                confidence=0.75,
                target_pixel=(256, 256),
                processing_time=50.0
            )
            mock_instance._update_tracking.return_value = None
            mock_instance._update_fps.return_value = 25.0
            mock_instance.enable_visualization = False
            mock_instance.safety_margin = 0.5
            mock_instance.get_performance_stats.return_value = {"fps": 25.0}
            mock_instance.get_segmentation_data.return_value = (None, None, None)
            mock.return_value = mock_instance
            
            return EnhancedUAVDetector(context="commercial", use_scallop=True)
    
    def _create_test_segmentation(self):
        """Create test segmentation map"""
        seg_map = np.zeros((512, 512), dtype=np.uint8)
        seg_map[200:300, 200:300] = 0  # Grass area
        seg_map[350:370, 350:370] = 1  # Small building
        return seg_map
    
    def _create_test_zones(self):
        """Create test landing zones"""
        return [
            {
                'center': (250, 250),
                'bbox': (200, 200, 100, 100),
                'area': 10000
            },
            {
                'center': (400, 400),
                'bbox': (375, 375, 50, 50),
                'area': 2500
            }
        ]
    
    def test_end_to_end_commercial_mission(self, integration_detector):
        """Test complete commercial mission workflow"""
        # Create test image
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Process frame
        result = integration_detector.process_frame(test_image, altitude=15.0)
        
        # Verify result structure
        assert hasattr(result, 'status')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'processing_time')
        
        # Get performance stats
        stats = integration_detector.get_enhanced_performance_stats()
        assert "context" in stats
        assert stats["context"] == "commercial"
    
    def test_context_adaptation_workflow(self, integration_detector):
        """Test workflow with context changes"""
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # Start with commercial
        result1 = integration_detector.process_frame(test_image, altitude=10.0)
        
        # Switch to emergency
        integration_detector.set_context("emergency")
        result2 = integration_detector.process_frame(test_image, altitude=10.0)
        
        # Switch to precision
        integration_detector.set_context("precision")
        result3 = integration_detector.process_frame(test_image, altitude=10.0)
        
        # All should succeed
        assert result1.status in ["TARGET_ACQUIRED", "NO_TARGET", "UNSAFE"]
        assert result2.status in ["TARGET_ACQUIRED", "NO_TARGET", "UNSAFE"]
        assert result3.status in ["TARGET_ACQUIRED", "NO_TARGET", "UNSAFE"]
        
        # Context should be updated
        assert integration_detector.context == "precision"
    
    def test_performance_under_load(self, integration_detector):
        """Test performance under repeated processing"""
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        processing_times = []
        
        # Process many frames
        for _ in range(20):
            start_time = time.time()
            result = integration_detector.process_frame(test_image, altitude=10.0)
            end_time = time.time()
            
            processing_times.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Each frame should process successfully
            assert result.status in ["TARGET_ACQUIRED", "NO_TARGET", "UNSAFE", "ERROR"]
        
        # Verify reasonable performance
        avg_time = np.mean(processing_times)
        assert avg_time < 1000  # Should be less than 1 second per frame
        
        # Get final stats
        stats = integration_detector.get_enhanced_performance_stats()
        assert "scallop_reasoning_count" in stats


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_scallop_failure_fallback(self):
        """Test fallback when Scallop fails"""
        with patch('enhanced_uav_detector.UAVLandingDetector') as mock_base:
            mock_instance = Mock()
            mock_instance._run_segmentation.return_value = np.zeros((256, 256), dtype=np.uint8)
            mock_instance._find_landing_zones.return_value = []
            mock_instance.get_segmentation_data.return_value = (None, None, None)
            mock_base.return_value = mock_instance
            
            detector = EnhancedUAVDetector(context="commercial", use_scallop=True)
            
            # Force Scallop to fail by mocking the reasoning engine
            with patch.object(detector, '_evaluate_zones_enhanced_scallop') as mock_scallop:
                mock_scallop.return_value = Mock(status="ERROR", confidence=0.0)
                
                # This should trigger fallback
                test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                result = detector.process_frame(test_image, altitude=10.0)
                
                # Should have fallen back successfully
                assert detector.fallback_count > 0
    
    def test_memory_constraints(self):
        """Test behavior under memory constraints"""
        with patch('enhanced_uav_detector.UAVLandingDetector') as mock_base:
            mock_instance = Mock()
            mock_instance._run_segmentation.return_value = np.zeros((1024, 1024), dtype=np.uint8)
            mock_instance.get_segmentation_data.return_value = (None, None, None)
            mock_base.return_value = mock_instance
            
            detector = EnhancedUAVDetector(context="commercial")
            
            # Test with large image
            large_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
            
            # Should handle gracefully
            result = detector.process_frame(large_image, altitude=10.0)
            assert result is not None
    
    def test_concurrent_access(self):
        """Test thread safety considerations"""
        # Note: This is a basic test - full thread safety would need more comprehensive testing
        with patch('enhanced_uav_detector.UAVLandingDetector'):
            detector = EnhancedUAVDetector(context="commercial")
            
            # Test state reset doesn't break detector
            detector.reset_state()
            detector.reset_state()  # Multiple resets should be safe
            
            # Test context switching
            detector.set_context("emergency")
            detector.set_context("commercial")
            detector.set_context("precision")
            
            assert detector.context == "precision"


# Pytest configuration and utilities
@pytest.fixture(scope="session")
def temp_model_file():
    """Create a temporary model file for testing"""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        # Create a minimal ONNX-like file (just for path testing)
        f.write(b"dummy onnx content")
        yield f.name
    
    # Cleanup
    Path(f.name).unlink()

@pytest.fixture(scope="session") 
def temp_config_file():
    """Create temporary configuration file"""
    config = {
        "model_path": "test_model.onnx",
        "contexts": {
            "commercial": {"threshold": 0.3, "safety_margin": 1.0},
            "emergency": {"threshold": 0.2, "safety_margin": 0.5},
            "precision": {"threshold": 0.5, "safety_margin": 1.5},
            "delivery": {"threshold": 0.35, "safety_margin": 0.8}
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as f:
        json.dump(config, f)
        yield f.name
    
    # Cleanup
    Path(f.name).unlink()

def pytest_configure(config):
    """Pytest configuration"""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")

def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "performance" in item.nodeid or "load" in item.nodeid:
            item.add_marker(pytest.mark.slow)

if __name__ == "__main__":
    # Run tests if script executed directly
    pytest.main([__file__, "-v"])
