# Segment 1: Motion Detection - Testing Guide

## Overview

This segment detects motion by comparing consecutive frames and alerting when significant pixel changes occur.

**Input Options:**
1. Webcam (live feed)
2. Video file (MP4, AVI, etc.)
3. Image sequence (folder of images)
4. Single image (looped/static)

---

## How Motion Detection Works

```
Previous Frame ──┐
                 ├──> Calculate Difference
Current Frame  ──┘
                 ↓
          Threshold to Binary (only large differences)
                 ↓
          Dilate (connect nearby changes)
                 ↓
          Find Contours (regions of change)
                 ↓
          Filter by Size (ignore tiny noise)
                 ↓
          Count Total Changed Pixels
                 ↓
          If > MOTION_THRESHOLD → ALERT!
```

---

## Configuration Parameters (Tunable)

These are in the code. Adjust based on your environment:

### `MOTION_THRESHOLD` (currently 2000)
- **Meaning:** Minimum pixels that must change to trigger alert
- **Too low (500):** Alerts on tiny movements, lots of false positives
- **Too high (5000):** Misses subtle motion, won't alert on distant objects
- **Sweet spot:** 1500-3000 depending on image size

### `MIN_CONTOUR_AREA` (currently 1000)
- **Meaning:** Minimum size of a motion object to consider
- **Too low (100):** Noise appears as motion
- **Too high (5000):** Misses small moving objects
- **Sweet spot:** 500-2000 depending on sensitivity needed

### `BLUR_KERNEL` (currently (21, 21))
- **Meaning:** How much to blur frames before comparing
- **Larger (31, 31):** Smoother, more tolerant, ignores small details
- **Smaller (11, 11):** Detects finer details, more sensitive to noise
- **Sweet spot:** (21, 21) is good general-purpose

---

## Testing Scenarios

### Scenario 1: Test with a Video File

**Create a test video:**
1. Record 10 seconds of yourself moving around with your phone
2. Save as `motion_test.mp4`
3. Run the script and choose option 2
4. Enter: `motion_test.mp4`

**Expected results:**
- ✅ Red alert border when you move
- ✅ Green boxes around your body
- ✅ Console shows: "MOTION DETECTED! Changed pixels: XXXX"
- ✅ Snapshots saved to `motion_events/` folder

---

### Scenario 2: Test with Image Sequence

**Create an image sequence:**
1. Take 3-4 similar photos of a scene
2. Move something (hand, object) between photos
3. Put all images in a folder: `test_images/`
4. Run the script and choose option 3
5. Enter: `test_images`

**Expected results:**
- ✅ First image: No motion (baseline)
- ✅ Second image: Motion detected where object moved
- ✅ Shows changed pixels count
- ✅ Cycles through images repeatedly

---

### Scenario 3: Test with Single Image (Stress Test)

This is interesting - a static image looped. Should produce almost no alerts:

1. Get any image (photo, screenshot, etc.)
2. Run the script and choose option 4
3. Enter image path

**Expected results:**
- ✅ No motion alerts (same frame repeated)
- ✅ Changed pixels ≈ 0
- ✅ Proves your threshold is working correctly

---

## What Each Alert Means

### 🟢 MONITORING (Green)
- System is running normally
- No significant motion detected
- Changed pixels < threshold

### 🔴 ALERT ACTIVE (Red)
- Motion detected!
- Red border appears on screen
- Message: "!!! MOTION ALERT !!!"
- Green boxes around detected motion regions
- Frame snapshot saved to `motion_events/`

---

## Console Output Explained

```
🚨 MOTION DETECTED! [2024-12-30 14:23:45]
   Frame: 45
   Changed pixels: 3500
   Objects detected: 1
```

- **Frame:** Which frame number this happened on
- **Changed pixels:** Total pixels that changed (compare to MOTION_THRESHOLD)
- **Objects detected:** How many motion regions found (filtered by MIN_CONTOUR_AREA)

---

## Debugging Tips

If motion isn't detected when it should be:

1. **Lower MOTION_THRESHOLD**
   - Currently: 2000
   - Try: 1000 or 1500
   - Too sensitive? Raise it back

2. **Lower MIN_CONTOUR_AREA**
   - Currently: 1000
   - Try: 500
   - Seeing noise? Raise it

3. **Enable debug windows** (uncomment in code):
   ```python
   cv2.imshow('Frame Difference', frame_diff)
   cv2.imshow('Thresholded Difference', thresh)
   cv2.imshow('Dilated Difference', dilated)
   ```
   This shows you exactly what the algorithm is seeing

4. **Check lighting**
   - Camera must be stationary
   - Lighting should be consistent
   - Shadows moving can trigger false alerts

---

## Output Files

### `motion_events/` folder
Contains:
- `motion_YYYYMMDD_HHMMSS_frameN.jpg` - Snapshots of detected motion
- `motion_log_YYYYMMDD_HHMMSS.txt` - Text log of all events

### Log File Example
```
======================================================================
MOTION DETECTION EVENT LOG
======================================================================

Total events: 3
Generated: 2024-12-30 14:23:50

----------------------------------------------------------------------
EVENTS:
----------------------------------------------------------------------

Event #1:
  Timestamp: 2024-12-30 14:23:10
  Frame: 45
  Changed pixels: 3500
  Objects detected: 1

Event #2:
  Timestamp: 2024-12-30 14:23:12
  Frame: 60
  Changed pixels: 4200
  Objects detected: 1

Event #3:
  Timestamp: 2024-12-30 14:23:15
  Frame: 87
  Changed pixels: 2800
  Objects detected: 2
```

---

## Keyboard Controls

While running:

- **`q`** - Quit and show summary
- **`s`** - Save event log to file
- **`d`** - Toggle debug views (if uncommented)

---

## Common Issues & Solutions

### "No motion detected even when moving"
**Solutions:**
- Lower MOTION_THRESHOLD to 1000-1500
- Lower MIN_CONTOUR_AREA to 500
- Check lighting (camera needs to see the movement clearly)
- Make bigger/faster movements

### "Too many false alerts"
**Solutions:**
- Raise MOTION_THRESHOLD to 3000-4000
- Raise MIN_CONTOUR_AREA to 2000
- Larger BLUR_KERNEL to (31, 31) to ignore small flickering
- Improve lighting consistency (avoid shadows, flickering lights)

### "Video file won't load"
**Solutions:**
- Check file format (MP4, AVI, MOV usually work)
- Check file path (use absolute path if relative doesn't work)
- Install ffmpeg: `pip install opencv-python-headless`
- Try converting video: use online converter or ffmpeg

### "Webcam not opening"
**Solutions:**
- Close other apps using camera
- Restart Python/script
- Try: `cap = cv2.VideoCapture(1)` instead of 0
- Check permission: System Preferences → Security → Camera

---

## Performance Notes

- **Video file:** Processing speed depends on resolution
  - 640x480: Very fast, near real-time
  - 1920x1080: Slower, but still workable
  - 4K: Slow, may drop frames

- **Image sequence:** No real-time constraint, processes at your speed

- **Memory:** Each snapshot saved = ~100-200KB file