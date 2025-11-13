# 💾 SAVE & LOAD GAME GUIDE

## How to Save and Load Your Game

---

## ✅ Save Functionality

### How to Save
1. Click the **"💾 Save"** button in the controls panel
2. You'll see a message: **"Game saved!"**
3. Your game is saved to `savegame.json`

### What Gets Saved
- Current day number
- Your balance
- All vehicles in your fleet
- Package delivery history
- Marketing level
- Complete game state

---

## ✅ Load Functionality (NEWLY ADDED!)

### How to Load
1. Click the **"📂 Load"** button in the controls panel
2. If a saved game exists:
   - You'll see: **"Game loaded successfully!"**
   - The game restores to your saved state
3. If no saved game:
   - You'll see: **"No saved game found!"**
   - Start a new game instead

### What Gets Loaded
- ✅ Day number restored
- ✅ Balance restored
- ✅ Fleet restored (all vehicles)
- ✅ Marketing level restored
- ✅ Game history restored
- ✅ UI reset to clean state

---

## 🎮 Button Location

### Controls Panel Layout
```
📦 Start Day        (full width)
🚚 Buy Vehicle      (full width)
🧠 Plan | 🔄 Clear  (split)
▶️ Execute | ⏭️ Next (split)
📈 Marketing        (full width)
💾 Save | 📂 Load   (split) ← HERE!
```

---

## 📂 Save File Location

### File Path
```
delivery_fleet_game/data/savegame.json
```

### File Contents
The save file stores your complete game state in JSON format:
- Game day
- Financial balance
- Fleet vehicles (with types and purchase days)
- Delivery history
- Marketing system state

---

## 🔄 Typical Workflow

### Save Your Progress
```
1. Play through several days
2. Build up your fleet
3. Reach a good financial position
4. Click "💾 Save"
5. Close the game
```

### Resume Later
```
1. Start the game
2. Click "📂 Load"
3. Continue from where you left off!
4. All your progress restored
```

---

## 💡 Tips

### When to Save
- ✅ After completing a successful day
- ✅ After purchasing vehicles
- ✅ After upgrading marketing
- ✅ Before trying risky strategies
- ✅ Before closing the game

### Multiple Playthroughs
- Game uses one save slot (`savegame.json`)
- Loading overwrites current progress
- Save file persists between sessions

### Automatic Save
- Game does NOT autosave
- You must click "💾 Save" manually
- Remember to save before quitting!

---

## 🔧 Technical Details

### Save Button (`on_save`)
```python
def on_save(self):
    """Save game."""
    self.engine.save_game()
    self.show_warning("Game saved!", Colors.TEXT_ACCENT)
```

### Load Button (`on_load`)
```python
def on_load(self):
    """Load saved game."""
    try:
        self.engine.load_game()
        # Reset UI state
        self.planned_routes = []
        self.planned_metrics = None
        self.package_status = {}

        # Clear planned metrics
        self.planned_cost_stat.set_value("$0", Colors.TEXT_SECONDARY)
        self.planned_revenue_stat.set_value("$0", Colors.TEXT_SECONDARY)
        self.planned_profit_stat.set_value("$0", Colors.TEXT_SECONDARY)

        # Reset button states
        self.buttons['plan_routes'].enabled = False
        self.buttons['clear'].enabled = False
        self.buttons['execute'].enabled = False
        self.buttons['next_day'].enabled = False

        # Update display
        self.update_stats()
        self.show_warning("Game loaded successfully!", Colors.PROFIT_POSITIVE)
    except FileNotFoundError:
        self.show_warning("No saved game found!", Colors.PROFIT_NEGATIVE)
    except Exception as e:
        self.show_warning(f"Error loading game!", Colors.PROFIT_NEGATIVE)
```

### What Gets Reset on Load
1. **Map State:**
   - Clears planned routes
   - Resets package status
   - Clears route metrics

2. **UI State:**
   - Disables action buttons
   - Clears cost/revenue/profit display
   - Updates all stats to loaded values

3. **Game State:**
   - Loads from `savegame.json`
   - Restores day, balance, fleet, history
   - Ready to continue playing

---

## 📊 Example Scenarios

### Scenario 1: Save After Good Day
```
Day 5 - Made $5,000 profit
Balance: $120,000
Fleet: 3 vehicles
Marketing: Level 2

1. Click "💾 Save"
2. Close game
3. Next time: Click "📂 Load"
4. Resume Day 5 with $120K and 3 vehicles!
```

### Scenario 2: No Saved Game
```
Fresh Install:
1. Click "📂 Load"
2. Message: "No saved game found!"
3. Click "📦 Start Day" to begin new game
```

### Scenario 3: Error Recovery
```
If loading fails:
1. Message: "Error loading game!"
2. Check console for details
3. Start new game or check savegame.json
```

---

## ⚠️ Important Notes

### Data Persistence
- Save file persists between game sessions
- Loading replaces current game state
- No "undo" after loading

### UI Reset
- Load clears any in-progress planning
- Route planning must be redone
- Packages for current day remain if day started

### Best Practices
1. **Save often** - Don't lose progress
2. **Load carefully** - Replaces current state
3. **Check day/balance** - Verify correct save loaded

---

## 🎯 Quick Reference

| Action | Button | Result |
|--------|--------|--------|
| **Save Game** | 💾 Save | Writes to savegame.json |
| **Load Game** | 📂 Load | Reads from savegame.json |
| **Check Save** | Look in data/ | See savegame.json file |

---

## 🚀 Now You Can:

✅ Save your progress anytime
✅ Load and resume later
✅ Experiment with strategies (save before risky moves)
✅ Close game without losing progress
✅ Continue multi-day campaigns

**Never lose your progress again!** 💾

---

## 📝 Summary

**The Load button is now working!**

- **Location:** Controls panel, next to Save
- **Function:** Restores saved game state
- **File:** data/savegame.json
- **Use:** Click "📂 Load" to resume

**Save often, play confidently!** 🎮
