import cv2
import numpy as np
from datetime import datetime
import os
from pathlib import Path

# ============================================================================
# PHASE 0: INPUT SOURCE SELECTION
# ============================================================================

print("="*70)
print("🎥 SURVEILLANCE SYSTEM - MOTION + SMOKE DETECTION (SEGMENT 3)")
print("="*70)
print("\n📹 Select input source:\n")
print("   1. Webcam (live feed)")
print("   2. Video file (MP4, AVI, MOV, etc.)")
print("   3. Image sequence (folder with images)")
print("   4. Single image (static, looped)")
print()

choice = input("Enter choice (1-4): ").strip()

cap = None
frame_count = 0
is_image_source = False
images = []
image_index = 0

if choice == "1":
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
    video_path = input("\n📁 Enter video file path: ").strip()
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
    print(f"\n📂 Loading {len(image_files)} images...")
    loaded_images = []
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is not None:
            loaded_images.append(img)
            print(f"   ✅ {os.path.basename(img_path)}")
    if not loaded_images:
        print(f"❌ Error: No images could be loaded")
        exit()
    images = loaded_images
    is_image_source = True
    source_name = f"Image sequence ({len(images)} images)"

elif choice == "4":
    image_path = input("\n📁 Enter image file path: ").strip()
    if not os.path.exists(image_path):
        print(f"❌ Error: File not found: {image_path}")
        exit()
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Could not load image!")
        exit()
    images = [img]
    is_image_source = True
    source_name = f"Image: {Path(image_path).name}"

else:
    print("❌ Invalid choice!")
    exit()

print(f"\n📊 Source: {source_name}\n")

# ============================================================================
# PHASE 1: CONFIGURATION
# ============================================================================

print("="*70)
print("⚙️  CONFIGURATION")
print("="*70)

# MOTION DETECTION parameters
MOTION_THRESHOLD = 2000
MIN_CONTOUR_AREA = 1000
BLUR_KERNEL = (21, 21)
DILATE_KERNEL = np.ones((5, 5), np.uint8)

# SMOKE DETECTION parameters
LOWER_SMOKE = np.array([0, 0, 200])
UPPER_SMOKE = np.array([180, 50, 255])
SMOKE_AREA_THRESHOLD = 5000
SMOKE_HISTORY_SIZE = 10
SMOKE_GROWTH_THRESHOLD = 1.3

print(f"\nMotion Detection:")
print(f"   Threshold: {MOTION_THRESHOLD} pixels")
print(f"   Min object: {MIN_CONTOUR_AREA} pixels")

print(f"\nSmoke Detection:")
print(f"   Min area: {SMOKE_AREA_THRESHOLD} pixels")
print(f"   Growth threshold: {SMOKE_GROWTH_THRESHOLD}x")

# Initialize frames
if not is_image_source:
    ret, previous_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read first frame!")
        exit()
else:
    previous_frame = images[0]
    ret = True

previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_gray = cv2.GaussianBlur(previous_gray, BLUR_KERNEL, 0)

# Tracking
smoke_area_history = []
events_log = []

print(f"\n{'='*70}")
print("🎯 SYSTEM READY - DUAL DETECTION ACTIVE")
print(f"{'='*70}")
print("\n🎬 Motion + Smoke Detection Running...")
print("   Press 'q' to quit, 's' to save log\n")

# ============================================================================
# HELPER FUNCTION
# ============================================================================

def save_event_log(events, output_dir='surveillance_events'):
    """Save unified event log"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'{output_dir}/surveillance_log_{timestamp}.txt'
    
    with open(log_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SURVEILLANCE SYSTEM - UNIFIED EVENT LOG\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total events: {len(events)}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("-"*70 + "\n")
        
        for i, event in enumerate(events, 1):
            f.write(f"\nEvent #{i}: {event['type']}\n")
            f.write(f"  Timestamp: {event['timestamp']}\n")
            f.write(f"  Frame: {event['frame']}\n")
            
            if event['type'] == 'MOTION':
                f.write(f"  Changed pixels: {event['changed_pixels']}\n")
                f.write(f"  Objects: {event['objects']}\n")
            elif event['type'] == 'FIRE':
                f.write(f"  Smoke area: {event['smoke_pixels']} px\n")
                f.write(f"  Growth rate: {event['growth_rate']:.2f}x\n")
    
    print(f"✅ Log saved: {log_file}")

# ============================================================================
# PHASE 2: MAIN DETECTION LOOP
# ============================================================================

try:
    while True:
        # Get frame
        if not is_image_source:
            ret, current_frame = cap.read()
            if not ret:
                print("\n✅ Stream ended")
                break
        else:
            current_frame = images[image_index].copy()
            image_index = (image_index + 1) % len(images)
        
        frame_count += 1
        display_frame = current_frame.copy()
        
        # ===================================================================
        # MOTION DETECTION
        # ===================================================================
        
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, BLUR_KERNEL, 0)
        
        frame_diff = cv2.absdiff(previous_gray, current_gray)
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, DILATE_KERNEL, iterations=2)
        
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_contours = []
        total_changed_pixels = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_CONTOUR_AREA:
                motion_contours.append(contour)
                total_changed_pixels += area
        
        motion_detected = total_changed_pixels > MOTION_THRESHOLD
        
        # ===================================================================
        # SMOKE DETECTION
        # ===================================================================
        
        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        smoke_mask = cv2.inRange(hsv, LOWER_SMOKE, UPPER_SMOKE)
        
        kernel = np.ones((5, 5), np.uint8)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
        smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE, kernel)
        
        smoke_pixels = np.sum(smoke_mask > 0)
        smoke_area_history.append(smoke_pixels)
        
        if len(smoke_area_history) > SMOKE_HISTORY_SIZE:
            smoke_area_history.pop(0)
        
        growth_rate = 1.0
        if len(smoke_area_history) >= 5:
            recent_avg = np.mean(smoke_area_history[-5:])
            old_avg = np.mean(smoke_area_history[:5])
            if old_avg > 0:
                growth_rate = recent_avg / old_avg
        
        fire_detected = smoke_pixels > SMOKE_AREA_THRESHOLD or \
                       (len(smoke_area_history) >= 5 and growth_rate > SMOKE_GROWTH_THRESHOLD)
        
        # ===================================================================
        # ALERT LOGIC - PRIORITY SYSTEM
        # ===================================================================
        
        # FIRE has priority over motion
        if fire_detected:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            events_log.append({
                'timestamp': timestamp,
                'frame': frame_count,
                'type': 'FIRE',
                'smoke_pixels': smoke_pixels,
                'growth_rate': growth_rate
            })
            
            print(f"🔥 FIRE ALERT! [{timestamp}] Frame {frame_count}")
            
            # Red/orange border + text
            cv2.rectangle(display_frame, (0, 0), 
                         (display_frame.shape[1], display_frame.shape[0]), 
                         (0, 100, 255), 25)
            cv2.putText(display_frame, "!!! FIRE EMERGENCY !!!", 
                       (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 100, 255), 5)
            
        elif motion_detected:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            events_log.append({
                'timestamp': timestamp,
                'frame': frame_count,
                'type': 'MOTION',
                'changed_pixels': int(total_changed_pixels),
                'objects': len(motion_contours)
            })
            
            print(f"⚠️  MOTION DETECTED [{timestamp}] Frame {frame_count}")
            
            # Yellow border + text (less severe than fire)
            cv2.rectangle(display_frame, (0, 0), 
                         (display_frame.shape[1], display_frame.shape[0]), 
                         (0, 255, 255), 20)
            cv2.putText(display_frame, "!!! MOTION ALERT !!!", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 4)
            
            # Draw motion boxes
            for i, contour in enumerate(motion_contours):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # ===================================================================
        # DISPLAY INFO (Always show status)
        # ===================================================================
        
        if fire_detected:
            status = "🔥 FIRE ALERT"
            status_color = (0, 100, 255)
        elif motion_detected:
            status = "⚠️  MOTION"
            status_color = (0, 255, 255)
        else:
            status = "🟢 MONITORING"
            status_color = (0, 255, 0)
        
        cv2.putText(display_frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
        
        # Stats
        cv2.putText(display_frame, f"Motion: {int(total_changed_pixels)}px | Smoke: {int(smoke_pixels)}px",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Frame: {frame_count}", 
                   (display_frame.shape[1] - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ===================================================================
        # SAVE SNAPSHOT
        # ===================================================================
        
        if motion_detected or fire_detected:
            os.makedirs('surveillance_events', exist_ok=True)
            timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            alert_type = "fire" if fire_detected else "motion"
            filename = f'surveillance_events/{alert_type}_{timestamp_file}_frame{frame_count}.jpg'
            cv2.imwrite(filename, display_frame)
        
        # ===================================================================
        # DISPLAY & INPUT
        # ===================================================================
        
        cv2.imshow('🎥 SURVEILLANCE SYSTEM - MOTION + SMOKE', display_frame)
        
        if is_image_source:
            delay_ms = 500
        else:
            delay_ms = 1
        
        key = cv2.waitKey(delay_ms) & 0xFF
        
        if key == ord('q'):
            print("\n⏹️  Stopping system...")
            break
        elif key == ord('s'):
            print("\n💾 Saving log...")
            save_event_log(events_log)
        
        previous_gray = current_gray

except KeyboardInterrupt:
    print("\n⏹️  Interrupted by user")

finally:
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("📊 FINAL REPORT")
    print("="*70)
    
    print(f"\nFrames processed: {frame_count}")
    print(f"Total events: {len(events_log)}")
    
    if events_log:
        motion_events = sum(1 for e in events_log if e['type'] == 'MOTION')
        fire_events = sum(1 for e in events_log if e['type'] == 'FIRE')
        
        print(f"\nEvent breakdown:")
        print(f"   Motion alerts: {motion_events}")
        print(f"   Fire alerts: {fire_events}")
        print(f"\nFirst event: {events_log[0]['timestamp']}")
        print(f"Last event: {events_log[-1]['timestamp']}")
        
        save_choice = input("\n💾 Save event log? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_event_log(events_log)
    
    if not is_image_source:
        cap.release()
    cv2.destroyAllWindows()
    
    print("\n✅ Surveillance system stopped")
    print("="*70)