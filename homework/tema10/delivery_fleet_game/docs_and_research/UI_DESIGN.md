# Pygame UI Design Document

## Design Philosophy

Create a clean, intuitive, and visually appealing interface that:
- Makes algorithmic concepts visible and understandable
- Provides smooth, responsive interactions
- Uses color and animation to enhance learning
- Maintains professional aesthetics

---

## Window Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  DELIVERY FLEET MANAGER          Day: 3  Balance: $125,430     │
├─────────────────────────────────────────────────────────────────┤
│                                                │                 │
│                                                │   GAME STATS    │
│                                                │                 │
│                                                │  Day: 3         │
│             MAP VIEW                           │  Balance: $XXX  │
│          (800x600 pixels)                      │  Fleet: 3 veh   │
│                                                │  Pending: 15    │
│         • Depot (center)                       │                 │
│         • Package destinations                 ├─────────────────┤
│         • Routes (colored lines)               │                 │
│         • Vehicles (animated)                  │  AGENT SELECT   │
│                                                │                 │
│                                                │  [Greedy]       │
│                                                │  [Backtrack]    │
│                                                │  [DP]           │
│                                                │                 │
│                                                │  [Compare All]  │
│                                                │                 │
├────────────────────────────────────────────────┼─────────────────┤
│                                                │                 │
│          METRICS PANEL                         │   CONTROLS      │
│                                                │                 │
│  Distance: 234.5 km  Cost: $187.60            │  [Start Day]    │
│  Revenue: $890.00    Profit: $702.40          │  [Plan Routes]  │
│  Packages: 15/15     Efficiency: 85%          │  [Execute Day]  │
│                                                │  [Next Day]     │
│                                                │  [Buy Vehicle]  │
└────────────────────────────────────────────────┴─────────────────┘

Total Window: 1200 x 800 pixels
- Map: 800 x 600 (left side)
- Sidebar: 400 x 800 (right side)
- Bottom metrics: 800 x 100
- Controls: 400 x 200
```

---

## Color Scheme

### Professional Delivery Theme

```python
COLORS = {
    # Background
    'bg_dark': (25, 30, 40),           # Dark blue-grey
    'bg_light': (245, 247, 250),        # Light grey-white
    'panel_bg': (45, 52, 65),           # Panel background

    # Map
    'depot': (255, 200, 0),             # Golden yellow (depot marker)
    'grid': (60, 70, 85),               # Subtle grid lines
    'map_bg': (35, 40, 50),             # Map background

    # Packages
    'package_pending': (100, 150, 255), # Blue (undelivered)
    'package_delivered': (50, 200, 100),# Green (delivered)
    'package_priority': (255, 100, 100),# Red (high priority)

    # Routes (different colors per vehicle)
    'route_colors': [
        (255, 100, 100),  # Red
        (100, 200, 255),  # Blue
        (100, 255, 150),  # Green
        (255, 200, 100),  # Orange
        (200, 100, 255),  # Purple
        (255, 255, 100),  # Yellow
    ],

    # Vehicles
    'vehicle_active': (255, 220, 0),    # Gold (moving)
    'vehicle_idle': (150, 150, 150),    # Grey (idle)

    # UI Elements
    'button_normal': (60, 120, 200),    # Blue button
    'button_hover': (80, 150, 230),     # Lighter blue
    'button_active': (40, 100, 180),    # Darker blue
    'button_disabled': (80, 85, 95),    # Grey

    'text_primary': (240, 245, 250),    # White-ish
    'text_secondary': (180, 190, 200),  # Light grey
    'text_accent': (100, 200, 255),     # Accent blue

    # Stats
    'profit_positive': (50, 200, 100),  # Green
    'profit_negative': (255, 80, 80),   # Red
    'neutral': (200, 200, 200),         # Grey
}
```

---

## Component Specifications

### 1. Map View (Primary Canvas)

**Size:** 800 x 600 pixels
**Location:** Left side of window

**Elements:**
- **Grid:** Subtle background grid (10km intervals)
- **Depot:** Large circular marker at center, always visible
- **Package Markers:** Small circles at destination points
  - Color-coded by status (pending/delivered/in-transit)
  - Size based on volume
  - Tooltip on hover showing package details
- **Routes:** Bezier curves or straight lines connecting stops
  - Different color per vehicle
  - Dashed line for planned routes
  - Solid line for executing routes
  - Arrow indicators showing direction
- **Vehicles:** Small truck/van icons
  - Animated movement along routes
  - Label showing vehicle ID
  - Trail effect showing path traveled

**Transformations:**
- World coordinates (0-100 km) → Screen coordinates (0-800 px)
- Centered view with padding
- Zoom capability (future enhancement)

---

### 2. Stats Panel (Top Right)

**Size:** 400 x 200 pixels

**Display:**
```
═══════════════════════════════
     GAME STATUS
═══════════════════════════════
Day:           3
Balance:       $125,430
Fleet:         3 vehicles
Pending:       15 packages

Today's Performance:
Distance:      234.5 km
Profit:        +$702.40
Delivery Rate: 100%
```

**Styling:**
- Clean box with border
- Icon for each stat type
- Color-coded values (green for positive, red for negative)

---

### 3. Agent Selector (Middle Right)

**Size:** 400 x 300 pixels

**Components:**
- Radio buttons for agent selection
- Description tooltip on hover
- "Compare All" button
- Last run metrics display

```
┌─────────────────────────────┐
│   SELECT ROUTING AGENT      │
├─────────────────────────────┤
│                             │
│ ○ Greedy                    │
│   Fast, O(n²)               │
│                             │
│ ○ Backtracking              │
│   Optimal, O(m^n)           │
│                             │
│ ○ Dynamic Programming       │
│   Medium, O(n²)             │
│                             │
│ [Compare All Agents]        │
│                             │
│ Last Run: Greedy            │
│ Profit: $702.40             │
└─────────────────────────────┘
```

---

### 4. Control Panel (Bottom Right)

**Size:** 400 x 200 pixels

**Buttons:**
- Large, clearly labeled
- Visual feedback on hover/click
- Disabled state when not applicable
- Icon + text labels

```
┌─────────────────────────────┐
│  GAME CONTROLS              │
├─────────────────────────────┤
│  [📦 Start Day]             │
│  [🧠 Plan Routes]           │
│  [▶️  Execute Day]           │
│  [⏭️  Next Day]              │
│  [🚚 Buy Vehicle]           │
│  [💾 Save Game]             │
└─────────────────────────────┘
```

---

### 5. Metrics Bar (Bottom)

**Size:** 800 x 100 pixels

**Real-time Metrics:**
- Horizontal bar showing key performance indicators
- Live updates during execution
- Color-coded based on performance

```
┌──────────────────────────────────────────────────────────────┐
│  📍 Distance: 234.5 km  │ 💰 Cost: $187.60  │ 📈 Revenue: $890│
│  💵 Profit: +$702.40    │ 📦 Packages: 15/15 │ ⚡ Eff: 85%   │
└──────────────────────────────────────────────────────────────┘
```

---

## Interaction Design

### User Flow

1. **Start Screen:**
   - Show menu: New Game / Load Game / Exit
   - Animated background (subtle)

2. **Main Game View:**
   - All panels visible
   - Start Day button highlighted

3. **Planning Phase:**
   - Select agent
   - Click "Plan Routes"
   - Routes appear on map
   - Metrics update

4. **Execution Phase:**
   - Click "Execute Day"
   - Vehicles animate along routes
   - Stats update in real-time
   - Packages change color as delivered

5. **End of Day:**
   - Summary overlay
   - Continue to next day or buy vehicle

### Mouse Interactions

- **Hover:** Tooltips for packages, buttons highlight
- **Click:** Button actions, select packages/vehicles
- **Drag:** Future: pan map view
- **Scroll:** Future: zoom map

### Keyboard Shortcuts

- `Space`: Execute current action
- `N`: Next day
- `S`: Save game
- `1-5`: Select agents
- `Esc`: Menu

---

## Animation Details

### Route Execution Animation

1. **Vehicle Movement:**
   - Smooth interpolation along route
   - Speed: ~100 km/h simulation (adjustable)
   - Rotation to face direction of travel

2. **Package Delivery:**
   - Pulse effect when vehicle reaches destination
   - Color change from blue → green
   - Particle effect (optional, subtle)

3. **Metrics Counter:**
   - Animated number counting up (revenue, distance)
   - Smooth transitions

### Transitions

- **Panel Slides:** Smooth in/out when showing comparisons
- **Fade Effects:** For overlays and messages
- **Button Feedback:** Scale slightly on click

---

## Fonts

```python
FONTS = {
    'title': ('Arial', 24, 'bold'),
    'heading': ('Arial', 18, 'bold'),
    'body': ('Arial', 14, 'normal'),
    'small': ('Arial', 12, 'normal'),
    'mono': ('Courier New', 14, 'normal'),  # For numbers/stats
}
```

---

## Implementation Priorities

### Phase 3A: Core Visualization (First)
1. Basic Pygame window setup
2. Map renderer with depot and packages
3. Simple route drawing
4. Basic UI panels (no fancy styling yet)
5. Integration with game engine

### Phase 3B: Interaction (Second)
1. Button components
2. Click handling
3. Agent selection
4. Execute day workflow

### Phase 3C: Polish (Third)
1. Colors and styling
2. Animations
3. Tooltips
4. Smooth transitions

---

This design ensures we build something beautiful and educational step by step!
