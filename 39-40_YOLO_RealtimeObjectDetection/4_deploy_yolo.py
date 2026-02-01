import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import threading
import queue
from pathlib import Path
import json
from datetime import datetime
import requests

# ==========================================
# PRODUCTION OPTIMIZATION: Speed & Efficiency
# ==========================================

class OptimizedYOLODetector:
    """
    Production-ready YOLO detector with optimizations

    Optimizations:
    1. Model quantization (FP16/INT8)
    2. Batch processing
    3. Frame skipping for video
    4. Multi-threading
    5. GPU memory management
    6. Async processing pipeline
    """

    def __init__(self, model_path, device='cuda:0', fp16=True):
        """
        Load optimized model

        device: 'cuda:0' for GPU, 'cpu' for CPU
        fp16: Use half-precision (2x faster, minimal accuracy loss)
        """
        self.model = YOLO(model_path)
        self.device = device
        self.fp16 = fp16

        # Warm up model (first inference is slow)
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy_input, device=device, half=fp16, verbose=False)

        print(f" Model loaded: {model_path}")
        print(f"   Device: {device}")
        print(f"   FP16: {fp16}")

    def detect_single(self, image, conf=0.5):
        """Single image detection with optimization"""
        results = self.model(
            image,
            device=self.device,
            half=self.fp16,
            conf=conf,
            verbose=False
        )[0]

        return results

    def detect_batch(self, images, conf=0.5, batch_size=8):
        """
        Batch processing for multiple images

        Processes multiple images simultaneously for efficiency
        Useful for: Processing video frames, bulk image analysis
        """
        detections = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]

            results = self.model(
                batch,
                device=self.device,
                half=self.fp16,
                conf=conf,
                verbose=False
            )

            detections.extend(results)

        return detections


# ==========================================
# VIDEO PROCESSING: Efficient Real-Time Detection
# ==========================================

class VideoStreamDetector:
    """
    Optimized video stream detection

    Techniques:
    - Frame skipping (process every Nth frame)
    - Async processing (detection doesn't block video)
    - Temporal smoothing (stable detections across frames)
    - Adaptive resolution (lower res when needed)
    """

    def __init__(self, model_path, skip_frames=2, buffer_size=4):
        self.detector = OptimizedYOLODetector(model_path)
        self.skip_frames = skip_frames
        self.frame_count = 0

        # Async processing
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.result_queue = queue.Queue(maxsize=buffer_size)

        # Detection smoothing
        self.detection_history = deque(maxlen=5)

        # Start async processor
        self.processing = True
        self.processor_thread = threading.Thread(target=self._async_processor)
        self.processor_thread.daemon = True
        self.processor_thread.start()

    def _async_processor(self):
        """Background thread for detection processing"""
        while self.processing:
            try:
                frame = self.frame_queue.get(timeout=1)
                results = self.detector.detect_single(frame)
                self.result_queue.put(results)
            except queue.Empty:
                continue

    def process_frame(self, frame):
        """
        Process video frame with optimizations

        Returns:
          - results: Detection results (or None if frame skipped)
          - processed: Whether this frame was processed
        """
        self.frame_count += 1

        # Frame skipping: Only process every Nth frame
        if self.frame_count % (self.skip_frames + 1) != 0:
            return None, False

        # Send frame for async processing
        if not self.frame_queue.full():
            self.frame_queue.put(frame)

        # Get results if available
        try:
            results = self.result_queue.get_nowait()
            self.detection_history.append(results)
            return results, True
        except queue.Empty:
            # Use previous results if new ones not ready
            if len(self.detection_history) > 0:
                return self.detection_history[-1], False
            return None, False

    def stop(self):
        """Clean shutdown"""
        self.processing = False
        self.processor_thread.join(timeout=2)


# ==========================================
# HEALTH APP: Production Workout Tracker
# ==========================================

class ProductionWorkoutTracker:
    """
    Production-grade workout tracking system

    Features:
    - Real-time detection with optimizations
    - Cloud data sync
    - Offline operation
    - Performance monitoring
    - Error handling & recovery
    """

    def __init__(self, model_path, api_endpoint=None):
        self.detector = VideoStreamDetector(model_path, skip_frames=1)
        self.api_endpoint = api_endpoint

        # Session management
        self.session = {
            'start_time': datetime.now(),
            'exercises_detected': [],
            'reps_counted': 0,
            'frames_processed': 0,
            'avg_fps': 0
        }

        # Offline queue (sync when connection available)
        self.offline_queue = []

    def log_exercise_rep(self, exercise_type, confidence):
        """Log exercise repetition"""
        rep_data = {
            'timestamp': datetime.now().isoformat(),
            'exercise': exercise_type,
            'confidence': confidence,
            'session_id': id(self.session)
        }

        self.session['reps_counted'] += 1
        self.session['exercises_detected'].append(rep_data)

        # Try to sync to cloud
        if self.api_endpoint:
            try:
                response = requests.post(
                    f"{self.api_endpoint}/workouts/log",
                    json=rep_data,
                    timeout=1
                )
                if response.status_code != 200:
                    self.offline_queue.append(rep_data)
            except:
                # Network error - queue for later
                self.offline_queue.append(rep_data)

    def sync_offline_data(self):
        """Sync queued data when connection available"""
        if not self.api_endpoint or len(self.offline_queue) == 0:
            return

        try:
            response = requests.post(
                f"{self.api_endpoint}/workouts/bulk_sync",
                json=self.offline_queue,
                timeout=5
            )

            if response.status_code == 200:
                print(f" Synced {len(self.offline_queue)} offline entries")
                self.offline_queue.clear()
        except Exception as e:
            print(f"Sync failed: {e}")

    def get_session_summary(self):
        """Generate workout session summary"""
        duration = (datetime.now() - self.session['start_time']).total_seconds()

        return {
            'duration_minutes': duration / 60,
            'total_reps': self.session['reps_counted'],
            'exercises': len(self.session['exercises_detected']),
            'avg_fps': self.session['avg_fps'],
            'offline_pending': len(self.offline_queue)
        }


# ==========================================
# FINANCE APP: Production Receipt Scanner
# ==========================================

class ProductionReceiptScanner:
    """
    Production receipt scanning and OCR pipeline

    Pipeline:
    1. YOLO detects receipt and components
    2. Extract text regions
    3. OCR (Tesseract/Google Vision)
    4. Parse amounts, dates, merchants
    5. Categorize expense
    6. Sync to database
    """

    def __init__(self, model_path, ocr_api_key=None):
        self.detector = OptimizedYOLODetector(model_path)
        self.ocr_api_key = ocr_api_key

        # Receipt processing queue
        self.processing_queue = []

    def scan_receipt(self, image):
        """
        Full receipt processing pipeline

        Returns structured receipt data
        """
        # Step 1: Detect receipt components
        results = self.detector.detect_single(image, conf=0.6)

        receipt_data = {
            'timestamp': datetime.now().isoformat(),
            'detections': [],
            'extracted_text': {},
            'parsed_fields': {}
        }

        # Step 2: Extract detected regions
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            bbox = box.xyxy[0].cpu().numpy()

            receipt_data['detections'].append({
                'class': class_name,
                'bbox': bbox.tolist(),
                'confidence': float(box.conf[0])
            })

            # Crop region for OCR
            x1, y1, x2, y2 = map(int, bbox)
            region = image[y1:y2, x1:x2]

            # Step 3: OCR on region
            text = self.ocr_region(region, class_name)
            receipt_data['extracted_text'][class_name] = text

        # Step 4: Parse structured fields
        receipt_data['parsed_fields'] = self.parse_receipt_fields(
            receipt_data['extracted_text']
        )

        return receipt_data

    def ocr_region(self, image_region, region_type):
        """
        OCR on image region

        Options:
        1. Tesseract (free, offline)
        2. Google Vision API (best accuracy, paid)
        3. AWS Textract (good for receipts, paid)
        """
        # Simplified - in production use actual OCR
        if self.ocr_api_key:
            # Use cloud OCR API
            return self._cloud_ocr(image_region)
        else:
            # Use local Tesseract
            return self._tesseract_ocr(image_region)

    def _tesseract_ocr(self, image):
        """Local Tesseract OCR"""
        try:
            import pytesseract
            text = pytesseract.image_to_string(image)
            return text.strip()
        except:
            return ""

    def _cloud_ocr(self, image):
        """Cloud OCR (Google Vision API example)"""
        # Placeholder - implement actual API call
        return "Mock OCR text"

    def parse_receipt_fields(self, extracted_text):
        """
        Parse structured fields from extracted text

        Uses regex + NLP to extract:
        - Total amount
        - Date
        - Merchant name
        - Payment method
        - Line items
        """
        import re

        fields = {
            'total': None,
            'date': None,
            'merchant': None,
            'category': None
        }

        # Simple parsing (in production use more sophisticated NLP)
        all_text = ' '.join(extracted_text.values())

        # Extract total amount
        amount_match = re.search(r'\$?(\d+\.\d{2})', all_text)
        if amount_match:
            fields['total'] = float(amount_match.group(1))

        # Extract date
        date_match = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', all_text)
        if date_match:
            fields['date'] = date_match.group(1)

        # Categorize based on merchant
        fields['category'] = self.categorize_expense(all_text)

        return fields

    def categorize_expense(self, text):
        """
        Auto-categorize expense based on merchant

        In production: Use ML classifier trained on historical data
        """
        text_lower = text.lower()

        categories = {
            'groceries': ['walmart', 'safeway', 'kroger', 'trader joe'],
            'dining': ['restaurant', 'cafe', 'pizza', 'burger'],
            'gas': ['shell', 'chevron', 'gas station', 'fuel'],
            'healthcare': ['pharmacy', 'cvs', 'walgreens', 'clinic'],
            'entertainment': ['cinema', 'movie', 'theater', 'netflix']
        }

        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category

        return 'other'


# ==========================================
# EDGE DEPLOYMENT: Mobile & Raspberry Pi
# ==========================================

class EdgeDeploymentGuide:
    """
    Guide for deploying YOLO on edge devices

    Edge Devices:
    - Smartphones (iOS/Android)
    - Raspberry Pi
    - NVIDIA Jetson
    - Intel NUC
    - Custom embedded systems
    """

    @staticmethod
    def export_for_mobile():
        """
        Export model for mobile deployment

        iOS: CoreML format
        Android: TensorFlow Lite
        """
        print("=" * 80)
        print("MOBILE DEPLOYMENT")
        print("=" * 80)
        print()

        model = YOLO('best.pt')

        # For iOS
        print("Exporting for iOS (CoreML)...")
        model.export(format='coreml')
        print(" iOS model: best.mlmodel")
        print()

        # For Android
        print("Exporting for Android (TFLite)...")
        model.export(format='tflite')
        print(" Android model: best.tflite")
        print()

        print("Integration:")
        print("  iOS: Use Vision framework or CoreML directly")
        print("  Android: Use TensorFlow Lite API")
        print()

    @staticmethod
    def raspberry_pi_setup():
        """
        Setup guide for Raspberry Pi deployment

        Requirements:
        - Raspberry Pi 4 (4GB+ RAM)
        - Camera module or USB webcam
        - Cooling (model runs hot)
        """
        print("=" * 80)
        print("RASPBERRY PI DEPLOYMENT")
        print("=" * 80)
        print()

        print("Step 1: Install Dependencies")
        print("  sudo apt-get update")
        print("  sudo apt-get install python3-opencv")
        print("  pip3 install ultralytics")
        print()

        print("Step 2: Optimize Model")
        print("Use YOLOv8n (nano) - smallest model")
        print("Reduce image size to 320�320")
        print("Skip frames (process every 3rd frame)")
        print("Expect ~10-15 FPS")
        print()

        print("Step 3: Run Detection")
        print("  python3 pi_detector.py")
        print()

    @staticmethod
    def jetson_deployment():
        """
        NVIDIA Jetson deployment (best for edge AI)

        Jetson devices:
        - Jetson Nano ($99): Entry level
        - Jetson Xavier NX ($399): Mid range
        - Jetson AGX Orin ($1,999): High performance
        """
        print("=" * 80)
        print("NVIDIA JETSON DEPLOYMENT")
        print("=" * 80)
        print()

        print("Advantages:")
        print("GPU acceleration (CUDA)")
        print("30-60 FPS real-time detection")
        print("TensorRT optimization (2-3x faster)")
        print("Low power consumption")
        print()

        print("Setup:")
        print("  1. Flash JetPack (includes CUDA, cuDNN)")
        print("  2. Install PyTorch with CUDA support")
        print("  3. Export model to TensorRT:")
        print()
        print("     model = YOLO('best.pt')")
        print("     model.export(format='engine')  # TensorRT")
        print()
        print("  4. Run with TensorRT backend:")
        print("     detector = YOLO('best.engine')")
        print("     results = detector(frame)  # 2-3x faster!")
        print()


# ==========================================
# MONITORING & ANALYTICS
# ==========================================

class ProductionMonitoring:
    """
    Production monitoring and analytics

    Metrics to track:
    - Inference time (latency)
    - FPS (throughput)
    - Model accuracy (confidence scores)
    - Error rates
    - Resource usage (CPU/GPU/memory)
    """

    def __init__(self):
        self.metrics = {
            'inference_times': deque(maxlen=1000),
            'fps_history': deque(maxlen=100),
            'detections_per_frame': deque(maxlen=1000),
            'confidence_scores': deque(maxlen=1000)
        }

    def log_inference(self, inference_time, num_detections, confidences):
        """Log inference metrics"""
        self.metrics['inference_times'].append(inference_time)
        self.metrics['detections_per_frame'].append(num_detections)
        self.metrics['confidence_scores'].extend(confidences)

    def get_performance_report(self):
        """Generate performance report"""
        if len(self.metrics['inference_times']) == 0:
            return {}

        inference_times = list(self.metrics['inference_times'])

        return {
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'p50_latency_ms': np.percentile(inference_times, 50) * 1000,
            'p95_latency_ms': np.percentile(inference_times, 95) * 1000,
            'p99_latency_ms': np.percentile(inference_times, 99) * 1000,
            'avg_fps': 1.0 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0,
            'avg_detections_per_frame': np.mean(list(self.metrics['detections_per_frame'])),
            'avg_confidence': np.mean(list(self.metrics['confidence_scores']))
        }


# ==========================================
# PRODUCTION DEPLOYMENT CHECKLIST
# ==========================================
#
# print("=" * 80)
# print("PRODUCTION DEPLOYMENT CHECKLIST")
# print("=" * 80)
# print()
# print(" Model Optimization:")
# print("   " Quantize to FP16 or INT8")
# print("   " Export to optimized format (ONNX/TensorRT)")
# print("   " Benchmark on target hardware")
# print()
# print(" Performance:")
# print("   " Implement frame skipping for video")
# print("   " Use async processing")
# print("   " Batch processing where possible")
# print("   " Monitor latency and FPS")
# print()
# print(" Reliability:")
# print("   " Handle camera disconnects gracefully")
# print("   " Offline operation + sync when online")
# print("   " Error logging and recovery")
# print("   " Health checks and monitoring")
# print()
# print(" Scalability:")
# print("   " Load balancing for multiple cameras")
# print("   " Cloud processing for heavy workloads")
# print("   " Edge processing for privacy/latency")
# print("   " Data pipeline: detection � processing � storage")
# print()
# print(" Security & Privacy:")
# print("   " Don't store sensitive video permanently")
# print("   " Encrypt data in transit")
# print("   " User consent for camera access")
# print("   " GDPR/compliance considerations")
# print()
#
# print("=" * 80)
# print("SESSIONS 39-40 COMPLETE!")
# print("=" * 80)
# print()
# print("<� YOU NOW UNDERSTAND:")
# print("   " Why YOLO revolutionized computer vision")
# print("   " How to use YOLOv8 for real-time detection")
# print("   " How to train custom models for YOUR needs")
# print("   " How to deploy in production at scale")
# print()
# print("=� NEXT STEPS:")
# print("   1. Collect images for YOUR use case")
# print("   2. Train custom model (health, finance, etc.)")
# print("   3. Deploy on edge device or cloud")
# print("   4. Build real-world app that helps people")
# print()
# print("=� THE INSIGHT:")
# print("   Computer vision is no longer magic.")
# print("   It's a tool you can use to solve real problems.")
# print("   The question is: What will YOU teach machines to see?")
# print()
