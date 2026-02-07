# Segment 2: Smoke Detection - Testing Guide

## Overview

This segment detects smoke by analyzing HSV color ranges and tracking if smoke area is **growing over time**.

**Key insight:** Smoke detection isn't just "is there smoke?" but "is smoke INCREASING?"

---

## How Smoke Detection Works

```
Current Frame
    ↓
Convert to HSV color space
    ↓
Create mask: Find all gray/white pixels
    (Smoke = low saturation + high brightness)
    ↓
Count smoke pixels in current frame
    ↓
Track smoke area over last 10 frames
    ↓
Calculate growth rate:
   Recent average / Old average = Growth ratio
    ↓
If growth > 1.3x (30% increase) → FIRE ALERT!
```

---

## Configuration Parameters

### HSV Smoke Range
```python
LOWER_SMOKE = np.array([0, 0, 200])      # H: any, S: low, V: high
UPPER_SMOKE = np.array([180, 50, 255])   # Gray/white detection
```

**What this means:**
- **H (0-180):** Any hue - smoke can be any color gray
- **S (0-50):** Low saturation - not very colorful (gray/white)
- **V (200-255):** High value - bright/light (smoke rises, catches light)

### `SMOKE_AREA_THRESHOLD` (currently 5000)
- **Meaning:** Minimum pixels that must be "smoke color" to trigger alert
- **Too low (1000):** Alerts on dust, reflections
- **Too high (10000):** Misses actual smoke
- **Sweet spot:** 3000-7000 depending on image size

### `SMOKE_HISTORY_SIZE` (currently 10)
- **Meaning:** How many frames to look back when calculating growth
- **Lower (5):** Quick response, but jittery
- **Higher (20):** Smooth detection, slower response
- **Sweet spot:** 8-15 frames

### `SMOKE_GROWTH_THRESHOLD` (currently 1.3)
- **Meaning:** How much smoke must grow to trigger alert
- **1.3** = 30% increase required
- **1.5** = 50% increase (stricter)
- **1.1** = 10% increase (very sensitive)

---

## Testing Scenarios

### Scenario 1: Test with White Paper (Safest)

**Simulate smoke with white/light gray paper:**

1. Get white or light gray paper
2. Slowly move it across the frame (simulates smoke rising/spreading)
3. Watch for alerts

**Expected results:**
- ✅ Alerts when paper is in frame
- ✅ Growth rate increases as paper moves
- ✅ Shows "SMOKE DETECTED" or "SMOKE EXPANDING"

**Why this works:** White paper has same color as smoke (high brightness, low saturation)

---

### Scenario 2: Test with Steam (Safe but Messy)

**Use actual steam (very realistic!):**

1. Boil water in a kettle or pot
2. Point camera at rising steam
3. Watch detection

**Expected results:**
- ✅ Steam detected as it rises
- ✅ Growth alerts as steam spreads
- Very realistic testing!

**Caution:** Don't point expensive cameras directly at steam

---

### Scenario 3: Test False Positives

**Try to fool the system:**

1. Shine bright white light into camera
2. Move white objects across frame
3. Try reflections

**Expected results:**
- Some false positives (this is normal)
- But growth rate check filters out most
- Pure white stationary objects won't trigger growth alert

---

### Scenario 4: Test with Your Images

Create images for smoke testing:
1. Image 1: Clean scene (no smoke)
2. Image 2: Faint white/gray area (light smoke)
3. Image 3: Larger white/gray area (more smoke)
4. Image 4: Even larger area (spreading smoke)

Put in folder, test with option 3 (image sequence)

**Expected results:**
- Frames 1→2: First smoke alert
- Frames 2→3: Growth detected
- Frames 3→4: More growth
- Growth rate increases over frames

---

## Console Output Explained

```
🔥 FIRE ALERT! [2024-12-30 14:23:45]
   Frame: 45
   Smoke area: 12500 pixels (15.2%)
   Growth rate: 1.45x
   Alert type: SMOKE EXPANDING
```

- **Frame:** Which frame number
- **Smoke area:** Pixels detected as smoke
- **Percentage:** What % of frame is smoke
- **Growth rate:** How much faster than baseline (1.45x = 45% growth)
- **Alert type:**
  - "SMOKE DETECTED" = Area > threshold
  - "SMOKE EXPANDING" = Growing rapidly

---

## Keyboard Controls

While running:

- **`q`** - Quit and show summary
- **`s`** - Save event log to file
- **`m`** - Toggle smoke mask display (shows what's detected as smoke)

---

## Debug: Viewing the Smoke Mask

Press **`m`** while running to toggle smoke mask display:

```
Smoke Mask window shows:
- White = detected as smoke
- Black = not smoke
```

This helps you understand if the HSV range is correct for your environment.

---

## Output Files

### `smoke_events/` folder
Contains:
- `fire_alert_YYYYMMDD_HHMMSS_frameN.jpg` - Alert screenshots
- `smoke_log_YYYYMMDD_HHMMSS.txt` - Event log

### Log File Example
```
======================================================================
SMOKE DETECTION EVENT LOG
======================================================================

Total events: 3
Generated: 2024-12-30 14:23:50

----------------------------------------------------------------------
EVENTS:
----------------------------------------------------------------------

Event #1:
  Timestamp: 2024-12-30 14:23:10
  Frame: 12
  Smoke area: 8500 pixels
  Smoke ratio: 0.12
  Growth rate: 1.35x
  Alert type: SMOKE EXPANDING

Event #2:
  Timestamp: 2024-12-30 14:23:12
  Frame: 28
  Smoke area: 15600 pixels
  Smoke ratio: 0.21
  Growth rate: 1.68x
  Alert type: SMOKE EXPANDING
```

---

## Common Issues & Solutions

### "No smoke detected even with white objects"
**Solutions:**
- Lower SMOKE_AREA_THRESHOLD to 2000-3000
- Check smoke mask (press `m`) - is white paper being detected?
- Increase brightness of the object
- Check HSV range - may need adjustment for your lighting

### "Too many false alerts"
**Solutions:**
- Raise SMOKE_AREA_THRESHOLD to 8000-10000
- Raise SMOKE_GROWTH_THRESHOLD to 1.5-1.7
- Avoid bright reflections/white objects in background
- Improve lighting consistency

### "Slow detection"
**Solutions:**
- Lower SMOKE_HISTORY_SIZE to 5 (quicker but jittery)
- Lower SMOKE_GROWTH_THRESHOLD to 1.1-1.2

### "Detects smoke but never "EXPANDING""
**Solutions:**
- Lower SMOKE_GROWTH_THRESHOLD to 1.2
- Make sure smoke actually grows in your test video
- Check growth rate values in console
