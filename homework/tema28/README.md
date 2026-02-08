# 🎥 OpenCV Surveillance & Fire Safety System - Complete Project

**Course:** Art of Programming - Python  
**Topic:** Computer Vision with OpenCV  
**Final Project:** Real-time surveillance system with motion and smoke detection

---

### **Exercises (Foundation)**

#### Exercise 1: Blue Object Detector
- **Concept:** Color-based object detection using HSV
- **Technique:** cv2.inRange() to create masks, cv2.findContours() to find objects
- **Result:** Detected blue shapes with bounding boxes and centroids

#### Exercise 2: Text Enhancement
- **Concept:** Adaptive vs Global thresholding for document processing
- **Technique:** cv2.adaptiveThreshold() with Gaussian and Mean methods
- **Result:** Enhanced readability of faded/damaged documents

#### Exercise 3: Contour Analysis
- **Concept:** Measuring and analyzing object properties
- **Techniques:** Area, perimeter, circularity, aspect ratio
- **Result:** Identified and classified shapes (circles vs rectangles)

### **Segments (Integration)**

#### Segment 1: Motion Detection ✅
- **Algorithm:** Frame differencing (comparing consecutive frames)
- **Key Step:** |Frame_t - Frame_t-1| > THRESHOLD → ALERT
- **Alert Color:** Red border + text
- **Use Case:** Intrusion detection, movement tracking

#### Segment 2: Smoke Detection ✅
- **Algorithm:** Color analysis + temporal growth tracking
- **Key Insight:** Smoke must be GROWING, not just present
- **Detection:** Low saturation (gray/white) + increasing area over time
- **Alert Color:** Orange-blue border + text
- **Use Case:** Fire/smoke detection safety system

#### Segment 3: Combined System ✅
- **Integration:** Both detectors running simultaneously
- **Priority System:** Fire alerts override motion alerts
- **Unified Logging:** Single event log for all alerts
- **Snapshots:** Save images of all detected events

---

## 🚀 How to Run the Project

### Prerequisites
```bash
pip install opencv-python numpy matplotlib
```

### Run Full Surveillance System (Segment 3)
```bash
python segment_3_combined_system.py
```

Then choose input source:
1. Webcam (live)
2. Video file (MP4, AVI)
3. Image sequence (folder)
4. Single image (test)

### Run Individual Segments
```bash
# Motion detection only
python segment_1_motion_detection.py

# Smoke detection only
python segment_2_smoke_detection.py
```

---

## 📊 System Architecture

```
Input Source (Webcam/Video/Images)
    ↓
├─→ MOTION DETECTOR
│   ├─ Frame differencing
│   ├─ Threshold binary image
│   ├─ Find contours
│   └─ Count changed pixels
│
├─→ SMOKE DETECTOR
│   ├─ Convert to HSV
│   ├─ Detect gray/white pixels
│   ├─ Track over last 10 frames
│   └─ Calculate growth rate
│
↓
Alert Priority Logic
├─ If FIRE → Red alert (priority)
├─ Else if MOTION → Yellow alert
└─ Else → Green (monitoring)
    
↓
Output
├─ Visual alert on screen
├─ Console log with timestamp
├─ Save snapshot image
└─ Log to event file
```

---

## ⚙️ Configuration Parameters

### Motion Detection
| Parameter | Value | Effect |
|-----------|-------|--------|
| `MOTION_THRESHOLD` | 2000 | Min pixels to trigger alert |
| `MIN_CONTOUR_AREA` | 1000 | Min object size (filters noise) |
| `BLUR_KERNEL` | (21, 21) | Smoothing (larger = less sensitive) |

### Smoke Detection
| Parameter | Value | Effect |
|-----------|-------|--------|
| `SMOKE_AREA_THRESHOLD` | 5000 | Min smoke pixels to alert |
| `SMOKE_HISTORY_SIZE` | 10 | Frames to track (for growth) |
| `SMOKE_GROWTH_THRESHOLD` | 1.3 | 30% growth needed to alert |
| `HSV_RANGE` | [0-180, 0-50, 200-255] | Gray/white detection |

---

## 🔧 Adjusting Parameters

### For More Sensitive Motion Detection
```python
MOTION_THRESHOLD = 1000  # Lower value = more sensitive
MIN_CONTOUR_AREA = 500   # Detect smaller objects
```

### For More Sensitive Smoke Detection
```python
SMOKE_AREA_THRESHOLD = 2000   # Lower = easier to detect
SMOKE_GROWTH_THRESHOLD = 1.1  # 10% growth instead of 30%
```

### For Faster Response (Trade-off: jittery)
```python
SMOKE_HISTORY_SIZE = 5  # Track fewer frames
```

### For Smoother Detection (Trade-off: slower response)
```python
SMOKE_HISTORY_SIZE = 20  # Track more frames
```

---

## 📁 Output Files

### Event Snapshots
```
surveillance_events/
├── motion_20240630_143045_frame123.jpg
├── fire_20240630_143102_frame456.jpg
└── motion_20240630_143210_frame789.jpg
```

### Event Log
```
surveillance_events/surveillance_log_20240630_143045.txt
```

Format:
```
EVENT #1: MOTION
  Timestamp: 2024-06-30 14:30:45
  Frame: 123
  Changed pixels: 3500
  Objects: 2

EVENT #2: FIRE
  Timestamp: 2024-06-30 14:31:02
  Frame: 456
  Smoke area: 12500 px
  Growth rate: 1.45x
```

---

## 🎨 Alert Visual System

### Status Indicators
- **🟢 Green** - Normal monitoring, no alerts
- **⚠️ Yellow border** - Motion detected (lower priority)
- **🔥 Red border** - Fire detected (higher priority, stops motion alerts)

### On-Screen Info
```
Status: 🔥 FIRE ALERT
Motion: 2500px | Smoke: 8500px
Frame: 456
```

---

## 🧪 Testing

### Test Scenarios Included
1. **Motion Detection:** Created 4-image sequence showing hand entering/leaving frame
2. **Smoke Detection:** Tested with realistic smoke/steam images
3. **Both Systems:** Run simultaneously with priority handling

### Test Results
- ✅ Motion detection works with 500ms delay (image sequences)
- ✅ Smoke detection triggers on gray/white pixels with growth tracking
- ✅ Both systems run in parallel without conflicts
- ✅ Fire alerts prioritized over motion alerts

---

## 🚨 Real-World Limitations & Future Improvements

### Current Limitations
- No network/email alerts (only console + snapshots)
- Single camera only (no multi-zone)
- No deep learning (uses traditional CV only)
- Affected by lighting changes

### Possible Enhancements (Not Implemented)
- [ ] Email/SMS alerts on fire detection
- [ ] Multi-zone sensitivity levels
- [ ] Person detection (exclude known residents)
- [ ] Thermal imaging for smoke detection
- [ ] Recording video (not just snapshots)
- [ ] Machine learning for pattern recognition
- [ ] Integration with smart home system