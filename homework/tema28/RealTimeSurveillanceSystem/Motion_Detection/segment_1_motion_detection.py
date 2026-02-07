import cv2
import numpy as np
from datetime import datetime
import os
from pathlib import Path

# ============================================================================
# PHASE 0: INPUT SOURCE SELECTION
# ============================================================================

print("="*70)
print("🎥 SURVEILLANCE SYSTEM - MOTION DETECTION (SEGMENT 1)")
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
        print("   Make sure your camera is connected and not in use.")
        exit()
    
    # Set camera properties
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
        print("   Check file format (MP4, AVI, MOV supported)")
        exit()
    
    # Get video properties
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
    
    # Find all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = sorted([
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.lower().endswith(image_extensions)
    ])
    
    if not images:
        print(f"❌ Error: No images found in {folder_path}")
        exit()
    
    # Load and validate images
    print(f"\n📂 Loading and validating {len(images)} images...")
    loaded_images = []
    for img_path in images:
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
    print(f"\n✅ Successfully loaded {len(images)}/{len(images)} images")
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
    
    # Store single image to loop
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

# Motion detection parameters
MOTION_THRESHOLD = 2000      # Minimum pixels that must change to trigger alert
MIN_CONTOUR_AREA = 1000      # Minimum object size (filters noise)
BLUR_KERNEL = (21, 21)       # Gaussian blur kernel (must be odd)
DILATE_KERNEL = np.ones((5, 5), np.uint8)  # Dilation kernel for morphology

print(f"\nMotion Detection Settings:")
print(f"   Motion threshold: {MOTION_THRESHOLD} pixels changed")
print(f"   Min object size: {MIN_CONTOUR_AREA} pixels")
print(f"   Blur kernel: {BLUR_KERNEL}")

# Initialize frame storage
if not is_image_source:
    ret, previous_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read first frame!")
        exit()
else:
    previous_frame = images[0]
    ret = True

# Prepare previous frame
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
previous_gray = cv2.GaussianBlur(previous_gray, BLUR_KERNEL, 0)

print(f"✅ Previous frame prepared: {previous_gray.shape}")

# Alert tracking
alarm_active = False
motion_detected_frames = 0
frame_counter = 0

# Event logging
events_log = []

print(f"\n{'='*70}")
print("🎯 SYSTEM READY")
print(f"{'='*70}")
print("\n📹 Running motion detection...")
print("   Press 'q' to quit")
print("   Press 's' to save event log")
print(f"\n{'='*70}\n")

# ============================================================================
# HELPER FUNCTIONS (Define before use)
# ============================================================================

def save_event_log(events, output_dir='motion_events'):
    """Save event log to a text file"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'{output_dir}/motion_log_{timestamp}.txt'
    
    with open(log_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MOTION DETECTION EVENT LOG\n")
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
            f.write(f"  Changed pixels: {event['changed_pixels']}\n")
            f.write(f"  Objects detected: {event['objects']}\n\n")
    
    print(f"✅ Event log saved to: {log_file}")

# ============================================================================
# PHASE 2: MAIN DETECTION LOOP
# ============================================================================

try:
    while True:
        # Get next frame
        if not is_image_source:
            ret, current_frame = cap.read()
            if not ret:
                print("\n✅ Video ended or stream closed")
                break
        else:
            # Cycle through images
            current_frame = images[image_index].copy()
            image_index = (image_index + 1) % len(images)
            ret = True
        
        frame_counter += 1
        
        # Prepare current frame
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, BLUR_KERNEL, 0)
        
        # ===================================================================
        # MOTION DETECTION ALGORITHM
        # ===================================================================
        
        # Step 1: Calculate difference between frames
        frame_diff = cv2.absdiff(previous_gray, current_gray)
        
        # Step 2: Apply threshold to get binary difference
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Step 3: Dilate to fill gaps and connect nearby changes
        dilated = cv2.dilate(thresh, DILATE_KERNEL, iterations=2)
        
        # Step 4: Find contours (connected regions of change)
        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Step 5: Filter contours by size and count changed pixels
        motion_contours = []
        total_changed_pixels = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Only consider contours above minimum size
            if area > MIN_CONTOUR_AREA:
                motion_contours.append(contour)
                total_changed_pixels += area
        
        # Step 6: Determine if motion was detected
        motion_detected = total_changed_pixels > MOTION_THRESHOLD
        
        # ===================================================================
        # ALERT HANDLING
        # ===================================================================
        
        # Copy frame for drawing
        display_frame = current_frame.copy()
        
        if motion_detected:
            alarm_active = True
            motion_detected_frames += 1
            
            # Log event
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            event_info = {
                'timestamp': timestamp,
                'frame': frame_counter,
                'changed_pixels': total_changed_pixels,
                'objects': len(motion_contours)
            }
            events_log.append(event_info)
            
            print(f"🚨 MOTION DETECTED! [{timestamp}]")
            print(f"   Frame: {frame_counter}")
            print(f"   Changed pixels: {total_changed_pixels}")
            print(f"   Objects detected: {len(motion_contours)}")
            
            # Draw red alert border
            cv2.rectangle(display_frame, (0, 0), 
                         (display_frame.shape[1], display_frame.shape[0]), 
                         (0, 0, 255), 20)
            
            # Draw alert text
            cv2.putText(display_frame, "!!! MOTION ALERT !!!", 
                       (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       2.5, (0, 0, 255), 4)
            
            # Draw bounding boxes around detected motion
            for i, contour in enumerate(motion_contours):
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Draw box
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), 
                            (0, 255, 0), 2)
                
                # Add label
                label = f"Motion #{i+1} ({int(area)}px)"
                cv2.putText(display_frame, label, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        else:
            alarm_active = False
            motion_detected_frames = 0
        
        # ===================================================================
        # DISPLAY INFORMATION
        # ===================================================================
        
        # Display frame number and status
        status_text = "🔴 ALERT ACTIVE" if motion_detected else "🟢 MONITORING"
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (0, 0, 255) if motion_detected else (0, 255, 0), 2)
        
        # Display changed pixel count
        pixel_text = f"Changed: {total_changed_pixels} px"
        cv2.putText(display_frame, pixel_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display frame counter
        frame_text = f"Frame: {frame_counter}"
        cv2.putText(display_frame, frame_text, 
                   (display_frame.shape[1] - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ===================================================================
        # SAVE SNAPSHOT ON ALERT
        # ===================================================================
        
        if motion_detected:
            # Create output directory if needed
            os.makedirs('motion_events', exist_ok=True)
            
            # Save snapshot
            timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'motion_events/motion_{timestamp_file}_frame{frame_counter}.jpg'
            cv2.imwrite(filename, display_frame)
        
        # ===================================================================
        # DISPLAY WINDOWS
        # ===================================================================
        
        # Main display
        cv2.imshow('🎥 Surveillance System - Motion Detection', display_frame)
        
        # Optional: Show difference map (for debugging)
        # Uncomment to see what changed between frames
        # cv2.imshow('Frame Difference', frame_diff)
        # cv2.imshow('Thresholded Difference', thresh)
        # cv2.imshow('Dilated Difference', dilated)
        
        # ===================================================================
        # USER INPUT HANDLING & FRAME DELAY
        # ===================================================================
        
        # Add delay for image sequences so they display at readable speed
        # For images: delay of 500ms (0.5 seconds per frame)
        # For video: delay of 1ms (let video play at native speed)
        if is_image_source:
            delay_ms = 200  # 500ms = 0.5 seconds per image - gives you time to see!
        else:
            delay_ms = 1    # 1ms for video (native speed)
        
        key = cv2.waitKey(delay_ms) & 0xFF
        
        if key == ord('q'):
            print("\n⏹️  Stopping surveillance system...")
            break
        
        elif key == ord('s'):
            print("\n💾 Saving event log...")
            save_event_log(events_log, 'motion_events')
        
        elif key == ord('d'):
            # Toggle debug views
            print("🔍 Debug mode toggled (uncomment cv2.imshow lines to use)")
        
        # Update previous frame for next iteration
        previous_gray = current_gray

except KeyboardInterrupt:
    print("\n⏹️  System interrupted by user")

finally:
    # ========================================================================
    # CLEANUP
    # ========================================================================
    
    print("\n" + "="*70)
    print("📊 SESSION SUMMARY")
    print("="*70)
    
    print(f"\nTotal frames processed: {frame_counter}")
    print(f"Total motion events: {len(events_log)}")
    
    if events_log:
        print(f"\nFirst alert: {events_log[0]['timestamp']}")
        print(f"Last alert: {events_log[-1]['timestamp']}")
        
        # Calculate motion statistics
        changed_pixels = [e['changed_pixels'] for e in events_log]
        print(f"\nMotion Statistics:")
        print(f"   Max change: {max(changed_pixels)} pixels")
        print(f"   Min change: {min(changed_pixels)} pixels")
        print(f"   Avg change: {np.mean(changed_pixels):.0f} pixels")
        
        # Ask to save
        save_choice = input("\n💾 Save event log? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_event_log(events_log, 'motion_events')
    
    # Release resources
    if not is_image_source:
        cap.release()
    cv2.destroyAllWindows()
    
    print("\n✅ Surveillance system stopped")
    print("="*70)