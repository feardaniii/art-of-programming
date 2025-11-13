# Delivery Fleet Management System

> An educational simulation game for learning algorithmic optimization through route planning and fleet management.

**Course:** Art of Programming - Advanced Algorithms
**Phase:** 1 - Foundation & Core Models (Console Version)
**Python Version:** 3.10+

---

## Overview

The Delivery Fleet Management System is an interactive simulation where you manage a delivery company's fleet and optimize package delivery routes. The game teaches fundamental algorithmic concepts including greedy algorithms, backtracking, dynamic programming, and other optimization techniques.

### Key Learning Objectives

- **Algorithm Design:** Implement and compare different routing algorithms
- **Object-Oriented Programming:** Work with clean class hierarchies and design patterns
- **Optimization:** Balance multiple objectives (cost, distance, capacity)
- **Problem Solving:** Apply algorithmic thinking to real-world scenarios
- **Software Architecture:** Understand separation of concerns and modular design

---

## Features

### Current (Phase 1)
✅ Core data models (Vehicle, Package, Route, Map, GameState)
✅ JSON-based data persistence
✅ Interactive console-based gameplay
✅ Day-by-day simulation cycle
✅ Financial management (balance, costs, revenues)
✅ Fleet management (purchase vehicles)
✅ Two routing algorithms:
  - **Greedy Agent:** Fast nearest-neighbor heuristic
  - **Backtracking Agent:** Exhaustive search with pruning
✅ Algorithm comparison and metrics
✅ Route validation and constraint checking

### Planned (Future Phases)
🔲 Additional algorithms (Dynamic Programming, Genetic, A*)
🔲 Graphical interface with Pygame
🔲 Animated route visualization
🔲 Advanced metrics and analytics
🔲 Multi-day planning
🔲 Time windows and priorities

---

## Installation

### Prerequisites
- Python 3.10 or higher
- No external dependencies for Phase 1!

### Setup
```bash
# Clone or download the project
cd delivery_fleet_game

# No pip install needed for Phase 1!
# All dependencies are from Python standard library

# Run the game
python main.py
```

---

## Quick Start

### 1. Start a New Game
```
$ python main.py

Choose option: 1 (Start New Game)
```

You'll start with:
- Balance: $100,000
- Fleet: 2 vehicles (1 small van, 1 medium truck)
- Day: 1

### 2. Begin a Day
```
Choose option: 4 (Start Day)
```

This loads packages for the current day from `data/packages_day1.json`.

### 3. Test Routing Agents
```
Choose option: 5 (Test Agent)
```

Compare different algorithms:
- **Greedy Agent:** O(n²), fast, decent solutions
- **Backtracking Agent:** O(m^n), slower, optimal solutions

The system will show metrics for each agent:
- Total distance
- Total cost
- Total revenue
- Profit
- Number of vehicles used

### 4. Apply Best Solution
```
Choose option: 6 (Apply Agent Solution)
```

Select the agent whose solution you want to use.

### 5. Execute the Day
```
Choose option: 7 (Execute Day)
```

This runs the delivery day with your planned routes and updates your balance.

### 6. Advance to Next Day
```
Choose option: 8 (Advance to Next Day)
```

### 7. Expand Your Fleet (Optional)
```
Choose option: 9 (Purchase Vehicle)
```

Buy new vehicles if your balance allows.

---

## Project Structure

```
delivery_fleet_game/
├── data/                        # Game data (JSON files)
│   ├── vehicles.json            # Vehicle type definitions
│   ├── map.json                 # Map configuration
│   ├── packages_day1.json       # Day 1 packages
│   ├── packages_day2.json       # Day 2 packages
│   └── initial_game_state.json  # Starting game state
│
├── src/
│   ├── models/                  # Data models
│   │   ├── vehicle.py           # Vehicle and VehicleType
│   │   ├── package.py           # Package model
│   │   ├── route.py             # Route model
│   │   ├── map.py               # DeliveryMap and Location
│   │   └── game_state.py        # GameState and DayHistory
│   │
│   ├── agents/                  # Routing algorithms
│   │   ├── base_agent.py        # Abstract base class
│   │   ├── greedy_agent.py      # Greedy algorithm
│   │   └── backtracking_agent.py# Backtracking algorithm
│   │
│   ├── core/                    # Game engine
│   │   ├── engine.py            # GameEngine (orchestration)
│   │   ├── router.py            # Routing utilities (TSP, etc.)
│   │   └── validator.py         # Constraint validation
│   │
│   └── utils/                   # Utilities
│       ├── data_loader.py       # JSON loading/saving
│       └── metrics.py           # Performance metrics
│
├── tests/                       # Unit tests (future)
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── DELIVERY_FLEET_SPECS.md      # Detailed specifications
└── ARCHITECTURE.md              # Technical architecture
```

---

## Game Mechanics

### Day Cycle

Each game day has three phases:

1. **Planning Phase:**
   - Load packages for the day
   - Test different routing algorithms
   - Compare solutions
   - Choose best approach

2. **Execution Phase:**
   - Apply chosen solution
   - Execute routes
   - Calculate costs and revenues
   - Update balance

3. **Management Phase:**
   - Review performance
   - Purchase new vehicles (optional)
   - Advance to next day

### Financial System

**Income:**
- Package delivery payments (varies by package)

**Expenses:**
- Vehicle operating costs (distance × cost_per_km)
- Vehicle purchase costs

**Goal:** Maximize long-term profit!

### Constraints

**Hard Constraints (Must satisfy):**
- Package volume ≤ Vehicle capacity
- Route distance ≤ Vehicle max range
- All packages assigned to exactly one vehicle
- Routes start and end at depot (0, 0)

**Soft Constraints (Optimization goals):**
- Minimize total distance
- Maximize profit (revenue - costs)
- Minimize number of vehicles used
- Maximize capacity utilization

### Win/Lose Conditions

**Win:**
- Reach day 30 with balance > $200,000
- Maintain positive daily profit for 10 consecutive days

**Lose:**
- Balance drops below $0 (bankruptcy)
- Three consecutive days of losses

---

## Routing Algorithms

### Greedy Agent

**Strategy:** Make locally optimal choices at each step

**Algorithm:**
1. Sort packages by value density (payment/volume)
2. Assign packages using First-Fit Decreasing bin packing
3. Optimize route order with Nearest Neighbor TSP
4. Optionally apply 2-opt local search

**Complexity:**
- Time: O(n²) where n = number of packages
- Space: O(n)

**Pros:**
- Very fast execution
- Simple to understand
- Good solutions for most cases

**Cons:**
- May not find optimal solution
- No backtracking
- Greedy choices can be suboptimal

### Backtracking Agent

**Strategy:** Exhaustive search with pruning

**Algorithm:**
1. Try assigning each package to each vehicle
2. Recursively assign remaining packages
3. Prune branches that violate capacity
4. Track best solution found (by profit)
5. Backtrack and try alternatives

**Complexity:**
- Time: O(m^n) where m = vehicles, n = packages
- Space: O(n) for recursion stack

**Pros:**
- Can find optimal solutions
- Systematic exploration
- Educational value

**Cons:**
- Exponential time complexity
- Only practical for ~15-20 packages
- Slow compared to greedy

### Pruning Backtracking Agent

Enhanced backtracking with:
- **Bounding:** Prune branches that can't beat best known solution
- **Symmetry breaking:** Avoid exploring equivalent solutions

Explores fewer nodes than basic backtracking.

---

## Example Gameplay Session

```
=== DELIVERY FLEET MANAGEMENT SYSTEM ===

Day 1 | Balance: $100,000.00 | Fleet: 2 vehicles | Pending: 10 packages

[Testing Greedy Agent...]
✓ Created 2 routes
Performance Metrics:
  Total Distance:        125.3 km
  Total Cost:           $87.20
  Total Revenue:        $705.00
  Total Profit:         $617.80
  Vehicles Used:             2
  Packages Delivered:       10

[Testing Backtracking Agent...]
✓ Explored 2048 nodes
✓ Created 2 routes
Performance Metrics:
  Total Distance:        118.7 km
  Total Cost:           $82.50
  Total Revenue:        $705.00
  Total Profit:         $622.50
  Vehicles Used:             2
  Packages Delivered:       10

Comparison:
Agent                    Profit      Distance    Vehicles    Packages
──────────────────────────────────────────────────────────────────
Greedy Agent           $617.80       125.3 km          2          10
Backtracking Agent     $622.50       118.7 km          2          10

Best Profit: Backtracking Agent ($622.50)

[Applying Backtracking Agent solution...]
[Executing day 1...]

=== Day 1 Complete ===
Delivered: 10/10 packages
Revenue: $705.00
Costs: $82.50
Profit: +$622.50
New Balance: $100,622.50
```

---

## Extending the System

### Adding a New Routing Agent

1. Create a new file in `src/agents/`:

```python
from .base_agent import RouteAgent
from ..models import Package, Vehicle, Route

class MyAgent(RouteAgent):
    def __init__(self, delivery_map):
        super().__init__(delivery_map, "My Agent")
        self.description = "My custom algorithm"

    def plan_routes(self, packages, fleet):
        # Your algorithm here
        routes = []
        # ... create routes ...
        return routes
```

2. Register in `main.py`:

```python
engine.register_agent("my_agent", MyAgent(engine.delivery_map))
```

3. Test and compare!

### Creating Custom Package Data

Create `data/packages_dayX.json`:

```json
{
  "day": X,
  "packages": [
    {
      "id": "pkg_xxx",
      "destination": {"x": 20, "y": 30},
      "volume_m3": 2.5,
      "payment": 50,
      "priority": 1,
      "description": "Electronics"
    }
  ]
}
```

---

## Development Roadmap

### Phase 1: Foundation ✅ (Current)
- Core models and game engine
- Console-based interface
- Greedy and backtracking algorithms
- Basic metrics and comparison

### Phase 2: Advanced Algorithms (Next)
- Dynamic Programming agent
- Genetic Algorithm agent
- A* search agent
- Enhanced metrics and visualization
- Unit test suite

### Phase 3: Graphical Interface
- Pygame-based UI
- Interactive map visualization
- Animated route execution
- Real-time statistics dashboard
- Enhanced user experience

---

## Educational Notes

### Design Patterns Used

1. **Strategy Pattern:** RouteAgent hierarchy allows swapping algorithms
2. **Data Class Pattern:** Clean, immutable data models
3. **Dependency Injection:** Map passed to agents for testability
4. **Repository Pattern:** DataLoader abstracts storage

### Algorithmic Concepts

- **Greedy Algorithms:** Local optimization, heuristics
- **Backtracking:** Exhaustive search, pruning, branch-and-bound
- **TSP (Traveling Salesman Problem):** NP-hard optimization
- **Bin Packing:** Capacity-constrained assignment
- **Constraint Satisfaction:** Validation and feasibility

### Code Quality

- Full type hints for IDE support
- Comprehensive docstrings
- Clean separation of concerns
- Testable, modular architecture
- PEP 8 compliant

---

## Troubleshooting

### "No packages found for day X"
Create `data/packages_dayX.json` following the template in existing day files.

### "Agent produced no routes"
Check that:
- Packages were loaded (option 4)
- Fleet has available vehicles
- Total package volume doesn't exceed fleet capacity

### Backtracking agent is very slow
This is expected! Backtracking is exponential. Either:
- Reduce `max_packages` parameter
- Use fewer packages in test data
- Use greedy agent for larger problems

---

## Contributing

This is an educational project. Students are encouraged to:

1. Implement new routing algorithms
2. Add optimization features
3. Create challenging test datasets
4. Improve metrics and analytics
5. Extend game mechanics

---

## License

This project is for educational purposes as part of the "Art of Programming" course.

---

## Acknowledgments

Created for teaching advanced algorithms and software design principles.
Demonstrates the application of theoretical concepts to practical problem-solving.

---

## Contact

For questions, issues, or improvements:
- Review `DELIVERY_FLEET_SPECS.md` for detailed specifications
- Review `ARCHITECTURE.md` for technical design details

---

**Happy optimizing! May your routes be short and your profits be high!** 🚚📦💰
