import torch
from PIL import Image
import numpy as np
from vehicle_detector import VehicleDetector
from vehicle_classifier import VehicleMakeModelClassifier
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark_models():
    try:
        # Initialize models
        detector = VehicleDetector()
        classifier = VehicleMakeModelClassifier()
        
        # Create test image
        test_image = Image.new('RGB', (640, 640), color='white')
        
        # Warm up
        logger.info("Warming up models...")
        for _ in range(10):
            detector.detect_vehicles(test_image)
            classifier.classify_vehicle(test_image)
        
        # Benchmark detection
        logger.info("Benchmarking detection...")
        times_detection = []
        for i in range(100):
            start = time.time()
            detector.detect_vehicles(test_image)
            times_detection.append(time.time() - start)
            if i % 20 == 0:
                logger.info(f"Detection progress: {i+1}/100")
        
        # Benchmark classification
        logger.info("Benchmarking classification...")
        times_classification = []
        for i in range(100):
            start = time.time()
            classifier.classify_vehicle(test_image)
            times_classification.append(time.time() - start)
            if i % 20 == 0:
                logger.info(f"Classification progress: {i+1}/100")
        
        # Report results
        det_avg = np.mean(times_detection)
        det_std = np.std(times_detection)
        cls_avg = np.mean(times_classification)
        cls_std = np.std(times_classification)
        
        logger.info("\nBenchmark Results:")
        logger.info(f"Detection: {det_avg:.3f}s ± {det_std:.3f}s")
        logger.info(f"Classification: {cls_avg:.3f}s ± {cls_std:.3f}s")
        
        # Verify against PDR requirements
        if det_avg + cls_avg > 0.25:
            logger.warning("⚠️ Combined latency exceeds PDR requirement of 250ms")
        else:
            logger.info("✅ Performance meets PDR requirements")
            
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == '__main__':
    benchmark_models() 