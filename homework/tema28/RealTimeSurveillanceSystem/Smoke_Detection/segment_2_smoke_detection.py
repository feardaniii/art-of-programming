import cv2
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================================
# PHASE 0: INPUT SOURCE SELECTION (Same as Segment 1)
# ============================================================================

print("="*70)
print("🎥 SURVEILLANCE SYSTEM - SMOKE DETECTION (SEGMENT 2)")
print("="*70)
print("\n📹 Select input source:\n")
print("   1. Webcam (live feed)")
print("   2. Video file (MP4, AVI, MOV, etc.)")
print("   3. Image sequence (folder with images)")
print("   4. Single image (static, looped)")
print()

choice = input("Enter choice (1-4): ").strip()

# Initialize capture source
cap = None
frame_count = 0
is_image_source = False
images = []
image_index = 0

if choice == "1":
    # Webcam
    print("\n🎥 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam!")
        exit()
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Webcam opened successfully")
    source_name = "Webcam"

elif choice == "2":
    # Video file
    video_path = input("\n📁 Enter video file path (e.g., video.mp4): ").strip()
    
    if not os.path.exists(video_path):
        print(f"❌ Error: File not found: {video_path}")
        exit()
    
    print(f"\n🎥 Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file!")
        exit()
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Video opened successfully")
    print(f"   FPS: {fps:.1f}, Frames: {total_frames}, Size: {width}×{height}")
    
    source_name = f"Video: {Path(video_path).name}"

elif choice == "3":
    # Image sequence from folder
    folder_path = input("\n📁 Enter folder path with images: ").strip()
    
    if not os.path.isdir(folder_path):
        print(f"❌ Error: Folder not found: {folder_path}")
        exit()
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = sorted([
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.lower().endswith(image_extensions)
    ])
    
    if not image_files:
        print(f"❌ Error: No images found in {folder_path}")
        exit()
    
    print(f"\n📂 Loading and validating {len(image_files)} images...")
    loaded_images = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"   ⚠️  Skipped (couldn't load): {os.path.basename(img_path)}")
        else:
            loaded_images.append(img)
            print(f"   ✅ Loaded: {os.path.basename(img_path)} - {img.shape}")
    
    if not loaded_images:
        print(f"\n❌ Error: No images could be loaded from {folder_path}")
        exit()
    
    images = loaded_images
    print(f"\n✅ Successfully loaded {len(images)}/{len(image_files)} images")
    is_image_source = True
    source_name = f"Image sequence ({len(images)} images)"

elif choice == "4":
    # Single image
    image_path = input("\n📁 Enter image file path: ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        exit()
    
    print(f"\n🎥 Loading image: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print("❌ Error: Could not load image!")
        exit()
    
    images = [img]
    is_image_source = True
    print(f"✅ Image loaded: {img.shape}")
    source_name = f"Image: {Path(image_path).name}"

else:
    print("❌ Invalid choice!")
    exit()

print(f"\n📊 Source: {source_name}\n")

# ============================================================================
# PHASE 1: CONFIGURATION & INITIALIZATION
# ============================================================================

print("="*70)
print("⚙️  CONFIGURATION")
print("="*70)

# Smoke detection parameters
# Smoke in HSV: Low saturation (gray/white), high value (brightness)
LOWER_SMOKE = np.array([0, 0, 200])      # Low hue/sat, high value
UPPER_SMOKE = np.array([180, 50, 255])   # Any hue, low sat, high value

SMOKE_AREA_THRESHOLD = 5000              # Minimum smoke pixels to alert
SMOKE_HISTORY_SIZE = 10                  # Track last N frames
SMOKE_GROWTH_THRESHOLD = 1.3             # 30% growth = fire alert

print(f"\nSmoke Detection Settings:")
print(f"   HSV Range: H[0-180], S[0-50], V[200-255]")
print(f"   Min smoke area: {SMOKE_AREA_THRESHOLD} pixels")
print(f"   History frames: {SMOKE_HISTORY_SIZE}")
print(f"   Growth threshold: {SMOKE_GROWTH_THRESHOLD}x (30% increase)")

# Initialize frame storage
if not is_image_source:
    ret, current_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read first frame!")
        exit()
else:
    current_frame = images[0]
    ret = True

print(f"✅ First frame loaded: {current_frame.shape}")

# Smoke tracking
smoke_area_history = []  # Stores smoke area for last N frames
events_log = []

print(f"\n{'='*70}")
print("🎯 SYSTEM READY")
print(f"{'='*70}")
print("\n📹 Running smoke detection...")
print("   Press 'q' to quit")
print("   Press 's' to save event log")
print("   Press 'm' to toggle smoke mask display")
print(f"\n{'='*70}\n")

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def save_event_log(events, output_dir='smoke_events'):
    """Save event log to a text file"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'{output_dir}/smoke_log_{timestamp}.txt'
    
    with open(log_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SMOKE DETECTION EVENT LOG\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total events: {len(events)}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("EVENTS:\n")
        f.write("-"*70 + "\n\n")
        
        for i, event in enumerate(events, 1):
            f.write(f"Event #{i}:\n")
            f.write(f"  Timestamp: {event['timestamp']}\n")
            f.write(f"  Frame: {event['frame']}\n")
            f.write(f"  Smoke area: {event['smoke_pixels']} pixels\n")
            f.write(f"  Smoke ratio: {event['smoke_ratio']:.2f}\n")
            f.write(f"  Growth rate: {event['growth_rate']:.2f}x\n")
            f.write(f"  Alert type: {event['alert_type']}\n\n")
    
    print(f"✅ Event log saved to: {log_file}")

# ============================================================================
# PHASE 2: MAIN DETECTION LOOP
# ============================================================================

show_mask = False  # Toggle for smoke mask display

try:
    while True:
        # Get next frame
        if not is_image_source:
            ret, current_frame = cap.read()
            if not ret:
                print("\n✅ Video ended or stream closed")
                break
        else:
            current_frame = images[image_index].copy()
            image_index = (image_index + 1) % len(images)
            ret = True
        
        frame_count += 1
        
        # ===================================================================
        # SMOKE DETECTION ALGORITHM
        # ===================================================================
        
        # Step 1: Convert to HSV color space
        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        
        # Step 2: Create smoke mask (detect gray/white pixels)
        # Smoke appears as low saturation (not colorful) + high value (bright)
        smoke_mask = cv2.inRange(hsv, LOWER_SMOKE, UPPER_SMOKE)
        
        # Step 3: Clean up mask with morphology
        kernel = np.ones((5, 5), np.uint8)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
        
        # Step 4: Calculate smoke area (number of white pixels in mask)
        smoke_pixels = np.sum(smoke_mask > 0)
        frame_total_pixels = current_frame.shape[0] * current_frame.shape[1]
        smoke_ratio = smoke_pixels / frame_total_pixels
        
        # Step 5: Track smoke over time
        smoke_area_history.append(smoke_pixels)
        
        # Keep only last N frames
        if len(smoke_area_history) > SMOKE_HISTORY_SIZE:
            smoke_area_history.pop(0)
        
        # Step 6: Calculate growth rate
        growth_rate = 1.0  # Default: no growth
        
        if len(smoke_area_history) >= 5:
            # Compare recent average to older average
            recent_avg = np.mean(smoke_area_history[-5:])
            old_avg = np.mean(smoke_area_history[:5])
            
            if old_avg > 0:
                growth_rate = recent_avg / old_avg
        
        # Step 7: Determine if fire detected
        fire_detected = False
        alert_type = "NONE"
        
        # Check 1: Smoke area exceeds threshold
        if smoke_pixels > SMOKE_AREA_THRESHOLD:
            fire_detected = True
            alert_type = "SMOKE DETECTED"
        
        # Check 2: Smoke is actively GROWING
        if len(smoke_area_history) >= 5 and growth_rate > SMOKE_GROWTH_THRESHOLD:
            fire_detected = True
            alert_type = "SMOKE EXPANDING"
        
        # ===================================================================
        # ALERT HANDLING
        # ===================================================================
        
        display_frame = current_frame.copy()
        
        if fire_detected:
            # Log event
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            event_info = {
                'timestamp': timestamp,
                'frame': frame_count,
                'smoke_pixels': smoke_pixels,
                'smoke_ratio': smoke_ratio,
                'growth_rate': growth_rate,
                'alert_type': alert_type
            }
            events_log.append(event_info)
            
            print(f"🔥 FIRE ALERT! [{timestamp}]")
            print(f"   Frame: {frame_count}")
            print(f"   Smoke area: {smoke_pixels} pixels ({smoke_ratio*100:.2f}%)")
            print(f"   Growth rate: {growth_rate:.2f}x")
            print(f"   Alert type: {alert_type}")
            
            # Draw blue alert border (different from motion red)
            cv2.rectangle(display_frame, (0, 0), 
                         (display_frame.shape[1], display_frame.shape[0]), 
                         (255, 100, 0), 20)  # Orange-blue (BGR)
            
            # Draw alert text
            cv2.putText(display_frame, "!!! FIRE ALERT !!!", 
                       (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       2.5, (255, 100, 0), 4)
        
        # ===================================================================
        # DISPLAY INFORMATION
        # ===================================================================
        
        # Status indicator
        status_text = "🔥 FIRE ALERT" if fire_detected else "🟢 MONITORING"
        status_color = (255, 100, 0) if fire_detected else (0, 255, 0)
        
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Smoke information
        smoke_text = f"Smoke: {int(smoke_pixels)}px ({smoke_ratio*100:.1f}%)"
        cv2.putText(display_frame, smoke_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Growth rate
        growth_text = f"Growth: {growth_rate:.2f}x"
        growth_color = (255, 100, 0) if growth_rate > SMOKE_GROWTH_THRESHOLD else (255, 255, 255)
        cv2.putText(display_frame, growth_text, (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, growth_color, 2)
        
        # Frame counter
        frame_text = f"Frame: {frame_count}"
        cv2.putText(display_frame, frame_text, 
                   (display_frame.shape[1] - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ===================================================================
        # SAVE SNAPSHOT ON ALERT
        # ===================================================================
        
        if fire_detected:
            os.makedirs('smoke_events', exist_ok=True)
            timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'smoke_events/fire_alert_{timestamp_file}_frame{frame_count}.jpg'
            cv2.imwrite(filename, display_frame)
        
        # ===================================================================
        # DISPLAY WINDOWS
        # ===================================================================
        
        cv2.imshow('🎥 Surveillance System - Smoke Detection', display_frame)
        
        # Optional: Show smoke mask
        if show_mask:
            # Convert mask to 3-channel for better visualization
            mask_display = cv2.cvtColor(smoke_mask, cv2.COLOR_GRAY2BGR)
            cv2.imshow('Smoke Mask (white=smoke)', mask_display)
        
        # ===================================================================
        # USER INPUT HANDLING & FRAME DELAY
        # ===================================================================
        
        if is_image_source:
            delay_ms = 2000
        else:
            delay_ms = 1
        
        key = cv2.waitKey(delay_ms) & 0xFF
        
        if key == ord('q'):
            print("\n⏹️  Stopping surveillance system...")
            break
        
        elif key == ord('s'):
            print("\n💾 Saving event log...")
            save_event_log(events_log, 'smoke_events')
        
        elif key == ord('m'):
            show_mask = not show_mask
            if show_mask:
                print("\n👁️  Smoke mask display: ON")
            else:
                print("\n👁️  Smoke mask display: OFF")
                cv2.destroyWindow('Smoke Mask (white=smoke)')

except KeyboardInterrupt:
    print("\n⏹️  System interrupted by user")

finally:
    # ========================================================================
    # CLEANUP & SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("📊 SESSION SUMMARY")
    print("="*70)
    
    print(f"\nTotal frames processed: {frame_count}")
    print(f"Total fire alerts: {len(events_log)}")
    
    if events_log:
        print(f"\nFirst alert: {events_log[0]['timestamp']}")
        print(f"Last alert: {events_log[-1]['timestamp']}")
        
        # Calculate statistics
        smoke_areas = [e['smoke_pixels'] for e in events_log]
        growth_rates = [e['growth_rate'] for e in events_log]
        
        print(f"\nSmoke Area Statistics:")
        print(f"   Max: {max(smoke_areas)} pixels")
        print(f"   Min: {min(smoke_areas)} pixels")
        print(f"   Avg: {np.mean(smoke_areas):.0f} pixels")
        
        print(f"\nGrowth Rate Statistics:")
        print(f"   Max: {max(growth_rates):.2f}x")
        print(f"   Min: {min(growth_rates):.2f}x")
        print(f"   Avg: {np.mean(growth_rates):.2f}x")
        
        # Count alert types
        expanding = sum(1 for e in events_log if e['alert_type'] == 'SMOKE EXPANDING')
        detected = sum(1 for e in events_log if e['alert_type'] == 'SMOKE DETECTED')
        
        print(f"\nAlert Types:")
        print(f"   Smoke detected: {detected}")
        print(f"   Smoke expanding: {expanding}")
        
        save_choice = input("\n💾 Save event log? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_event_log(events_log, 'smoke_events')
    
    # Release resources
    if not is_image_source:
        cap.release()
    cv2.destroyAllWindows()
    
    print("\n✅ Surveillance system stopped")
    print("="*70)