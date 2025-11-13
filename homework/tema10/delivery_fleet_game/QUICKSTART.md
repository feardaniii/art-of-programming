# Quick Start Guide

Get the Delivery Fleet Manager running in 60 seconds!

---

## Installation

### Step 1: Install Pygame

```bash
cd delivery_fleet_game
pip install -r requirements.txt
```

This installs pygame (the only dependency).

### Step 2: Run the Game!

Choose your interface:

**🎮 Graphical Version (Recommended):**
```bash
python3 main_pygame.py
```

**💻 Console Version (Text-based):**
```bash
python3 main.py
```

---

## First Playthrough (GUI Version)

### 1. Window Opens
You'll see:
- **Map** (left): Shows depot at center, grid background
- **Stats Panel** (top right): Day 1, $100,000 balance, 2 vehicles
- **Agent Panel** (middle right): Choose routing algorithm
- **Controls** (bottom right): Action buttons

### 2. Click "📦 Start Day"
- Loads 10 packages for Day 1
- Blue dots appear on map (package destinations)
- "Plan Routes" button becomes active

### 3. Select an Agent
- Click a radio button:
  - **Greedy (Fast)** ← Start with this!
  - **Greedy + 2-opt** (better routes, slower)
  - **Backtracking** (optimal, much slower)
  - **Pruning Backtrack** (smart search)

### 4. Click "🧠 Plan Routes"
- Agent calculates optimal routes
- Colored lines appear showing delivery paths
- Console shows profit calculation

### 5. Click "▶️ Execute"
- Routes are executed
- Balance updates with profit
- Packages turn green (delivered)

### 6. Click "⏭️ Next"
- Advances to Day 2
- Repeat!

---

## Understanding the Map

```
🟡 Yellow circle = Depot (your home base)
🔵 Blue dots = Packages to deliver
🟢 Green dots = Delivered packages
━━ Colored lines = Delivery routes
□ Small rectangles = Vehicles
```

---

## Tips for Success

### 💡 Algorithm Choice

**Use Greedy when:**
- Many packages (15+)
- Need fast results
- Learning the basics

**Use Backtracking when:**
- Few packages (< 12)
- Want optimal solution
- Studying algorithm behavior

**Use Pruning Backtracking when:**
- Medium packages (12-15)
- Want better than greedy
- Have a few seconds to wait

### 💰 Financial Strategy

1. **Start conservative:** Use greedy agent, learn the game
2. **Expand wisely:** Buy vehicles when balance > $150,000
3. **Monitor profit:** Green numbers = good, red = trouble
4. **Optimize routes:** Try different agents, compare profits

### 🚚 Fleet Management

**Small Van** ($15,000):
- Capacity: 10m³
- Cost: $0.50/km
- Good for: Starting out, small deliveries

**Medium Truck** ($35,000):
- Capacity: 25m³
- Cost: $0.80/km
- Good for: Growth phase, mixed loads

**Large Truck** ($65,000):
- Capacity: 50m³
- Cost: $1.20/km
- Good for: Bulk deliveries, late game

---

## Keyboard Shortcuts

- `Space` = Plan routes (if enabled)
- `Esc` = Quit game

---

## Troubleshooting

### "No packages found for day X"
- Days 1-2 have package data
- Create more days by copying `data/packages_day1.json` structure

### Routes look weird
- This is normal! Algorithms optimize for profit, not visual beauty
- Try different agents to compare approaches

### Backtracking is slow
- Expected behavior - it's exploring all possibilities
- Reduce `max_packages` in code or use fewer packages

### Window doesn't open
- Make sure pygame is installed: `pip install pygame`
- Check Python version: `python3 --version` (need 3.10+)

---

## What to Try

### Experiment 1: Algorithm Comparison
1. Start Day 1
2. Test all 4 agents (don't execute yet)
3. Compare console output for profit/distance
4. Which is best? Why?

### Experiment 2: Fleet Optimization
1. Play until Day 5
2. Buy a large truck
3. See how routes change with more capacity
4. Is it worth the cost?

### Experiment 3: Custom Packages
1. Edit `data/packages_day1.json`
2. Add a huge package (20m³)
3. See how algorithms handle it
4. What strategy emerges?

---

## Next Steps

Once comfortable:
1. Read `DELIVERY_FLEET_SPECS.md` for game mechanics
2. Study `ARCHITECTURE.md` for code structure
3. Implement your own routing agent!
4. Create challenging test scenarios

---

## Learning Goals

This project teaches:
- **Greedy Algorithms:** Fast, local optimization
- **Backtracking:** Exhaustive search with pruning
- **Trade-offs:** Speed vs. optimality
- **Constraint Satisfaction:** Capacity, range limits
- **TSP (Traveling Salesman):** Classic optimization problem
- **Software Architecture:** Clean code design

---

**Have fun optimizing! 🚚📦💰**

Questions? Check the main README.md or review the code comments.
