# Student Guide: Building Your Own Routing Agent

This guide will walk you through the process of creating, implementing, and testing your own routing algorithm for the Delivery Fleet Game.

---

## Table of Contents

1. [Phase 1: Understanding the Codebase](#phase-1-understanding-the-codebase)
2. [Phase 2: Algorithm Design & Research](#phase-2-algorithm-design--research)
3. [Phase 3: Implementation](#phase-3-implementation)
4. [Phase 4: Integration & Testing](#phase-4-integration--testing)
5. [Phase 5: Iteration & Improvement](#phase-5-iteration--improvement)

---

## Phase 1: Understanding the Codebase

### Step 1.1: Identify the Project Structure

Start by exploring the project directory:

```
delivery_fleet_game/
├── main_pygame.py              # GUI entry point - WHERE YOU'LL REGISTER YOUR AGENT
├── src/
│   ├── agents/
│   │   ├── base_agent.py       # TEMPLATE you must inherit from
│   │   ├── greedy_agent.py     # Example implementation
│   │   └── backtracking_agent.py
│   ├── core/                   # Game engine (don't modify)
│   ├── models/                 # Data structures (Package, Vehicle, Route)
│   └── utils/                  # Helper functions
└── data/                       # Game configuration files
```

**Key Files to Study:**
- `src/agents/base_agent.py` - Your template
- `src/agents/greedy_agent.py` - Reference implementation
- `main_pygame.py` (lines 192-198) - Agent registration
- `src/models/__init__.py` - Understanding Package, Vehicle, Route, DeliveryMap

### Step 1.2: Study the Base Agent Template

Open and read `src/agents/base_agent.py`:

**Key Points:**
```python
class RouteAgent(ABC):
    def __init__(self, delivery_map: DeliveryMap, name: str = "Base Agent"):
        self.delivery_map = delivery_map  # Use for distance calculations
        self.name = name
        self.description = "Base routing agent"

    @abstractmethod
    def plan_routes(self, packages: List[Package], fleet: List[Vehicle]) -> List[Route]:
        """YOU MUST IMPLEMENT THIS METHOD"""
        raise NotImplementedError()

    # Helper methods provided:
    def validate_inputs(self, packages, fleet) -> bool:
        """Checks if inputs are valid"""

    def calculate_metrics(self, routes) -> Dict:
        """Calculates performance metrics"""
```

**What Your Agent Must Do:**
1. Inherit from `RouteAgent`
2. Implement `plan_routes(packages, fleet)` method
3. Return a list of `Route` objects
4. Respect vehicle capacity constraints
5. Use `self.delivery_map` for distance calculations

### Step 1.3: Understand the Data Models

**⚠️ CRITICAL - Most Common Student Errors:**

| ❌ WRONG (causes errors) | ✅ CORRECT |
|-------------------------|-----------|
| `route.current_load` | `route.total_volume` |
| `route.capacity` | `route.vehicle.vehicle_type.capacity_m3` |
| `route.load` | `route.total_volume` |
| Not adding RadioButton | Add to `self.agent_radios` in main_pygame.py |
| Return `None` | Always return `[]` for empty lists |

---

**Package** (`src/models/package.py`):
```python
class Package:
    id: str
    destination: tuple[float, float]  # (x, y) coordinates
    volume_m3: float                  # Size in cubic meters
    payment: float                    # Revenue for delivery
    value_density: float              # payment / volume_m3
```

**Vehicle** (`src/models/vehicle.py`):
```python
class Vehicle:
    vehicle_id: str
    vehicle_type: VehicleType         # Has capacity_m3, daily_cost
    current_location: tuple           # Current position

class VehicleType:
    capacity_m3: float                # Maximum cargo capacity
    daily_cost: float                 # Cost to use per day
```

**Route** (`src/models/route.py`):
```python
class Route:
    vehicle: Vehicle                  # Assigned vehicle
    packages: List[Package]           # Packages to deliver
    stops: List[tuple]                # Ordered delivery locations
    delivery_map: DeliveryMap         # For distance calculations

    def add_package(self, package) -> bool:
        """Try to add package, returns False if capacity exceeded"""

    # Read-only properties (calculated automatically):
    @property
    def total_distance(self) -> float:
        """Total distance traveled (depot -> stops -> depot)"""

    @property
    def total_volume(self) -> float:
        """Total volume of all packages (sum of package.volume_m3)"""

    @property
    def total_cost(self) -> float:
        """Total cost (vehicle daily cost)"""

    @property
    def total_revenue(self) -> float:
        """Total revenue (sum of package payments)"""
```

**⚠️ IMPORTANT - Common Mistake:**
- Use `route.total_volume` NOT `route.current_load` (doesn't exist!)
- Properties are read-only, calculated automatically
- Don't try to set them directly: `route.total_volume = 100` won't work

**DeliveryMap** (`src/models/delivery_map.py`):
```python
class DeliveryMap:
    depot: tuple[float, float]        # Starting point (0, 0)

    def distance(self, loc1, loc2) -> float:
        """Calculate Euclidean distance between two locations"""
```

### Step 1.4: Study a Reference Implementation

Read `src/agents/greedy_agent.py` carefully:

**Algorithm Flow:**
1. Sort packages by value density (greedy choice)
2. Assign packages to vehicles (First-Fit Decreasing)
3. Optimize each route using Nearest Neighbor TSP
4. Optionally apply 2-opt improvement

**Key Takeaways:**
- Use `self.validate_inputs()` before processing
- Create `Route` objects with vehicle and packages
- Use helper classes like `Router` for TSP optimization
- Print progress messages for debugging
- Handle cases with insufficient capacity

### Step 1.5: Identify GUI Integration Points

Open `main_pygame.py` and find the `_register_agents()` method (lines 192-198):

```python
def _register_agents(self):
    """Register all routing agents."""
    self.engine.register_agent("greedy", GreedyAgent(self.engine.delivery_map))
    self.engine.register_agent("greedy_2opt", GreedyAgent(self.engine.delivery_map, use_2opt=True))
    # ... more agents
```

**What You'll Need to Do:**
1. Import your agent class at the top of `main_pygame.py`
2. Add one line in `_register_agents()` to register your agent
3. Choose a unique identifier string (e.g., "student", "my_algorithm")

---

## Phase 2: Algorithm Design & Research

### Step 2.1: Understand the Problem

This is a **Vehicle Routing Problem (VRP)** with:
- **Objective:** Maximize profit (revenue - costs)
- **Constraints:**
  - Vehicle capacity limits
  - Must start and end at depot
  - Each package delivered exactly once
- **Metrics to Optimize:**
  - Total distance (less is better - reduces time)
  - Number of vehicles used (less is better - reduces cost)
  - Package revenue (more is better)

**Key Trade-offs:**
- Speed vs. Optimality (fast heuristics vs. exhaustive search)
- Simplicity vs. Sophistication
- Greedy vs. Look-ahead strategies

### Step 2.2: Brainstorm on Paper

Grab pen and paper and sketch out ideas:

**Questions to Consider:**
1. How do I group packages into vehicle loads?
   - By proximity? By value? By size?
   - Can I cluster nearby destinations?

2. How do I order stops within a route?
   - Nearest neighbor from depot?
   - Create a mini traveling salesman tour?
   - Priority by package value?

3. What makes a "good" solution?
   - Short total distance?
   - Few vehicles used?
   - High-value packages prioritized?

**Possible Approaches to Explore:**

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Random** | Randomly assign packages and routes | Very fast, baseline | Very poor quality |
| **Nearest Depot** | Always deliver to closest location first | Simple | Ignores capacity & efficiency |
| **Greedy by Value** | Prioritize high-value packages | Good revenue | May waste capacity |
| **Clustering** | Group nearby packages, one vehicle per cluster | Efficient routes | Complex to implement |
| **Savings Algorithm** | Clarke-Wright savings heuristic | Good balance | Moderate complexity |
| **Sweep Algorithm** | Polar sweep from depot | Simple, decent results | Works best for circular distributions |
| **Genetic Algorithm** | Evolutionary optimization | Can find good solutions | Slow, complex |
| **Branch & Bound** | Exhaustive with pruning | Optimal (for small instances) | Exponential complexity |

### Step 2.3: Research Techniques

**Use These Resources:**
1. **Wikipedia:** Search for "Vehicle Routing Problem", "Traveling Salesman Problem"
2. **YouTube:** Look for VRP algorithm visualizations
3. **ChatGPT/Claude:** Ask about specific algorithms
   - "Explain Clarke-Wright Savings Algorithm for VRP"
   - "What is the Sweep Algorithm?"
   - "How does Nearest Neighbor TSP work?"

**Document Your Research:**
Create a simple markdown file with your findings:

```markdown
# My Algorithm Research Notes

## Algorithm Chosen: [Name]

### How It Works:
1. Step 1...
2. Step 2...

### Why I Chose This:
- Reason 1
- Reason 2

### Expected Time Complexity: O(?)

### Expected Performance:
- Distance: [Better/Worse than Greedy?]
- Speed: [Fast/Medium/Slow?]
```

### Step 2.4: Test Your Logic with Pseudocode

Before coding, write pseudocode:

```
FUNCTION plan_routes(packages, fleet):
    // Validate inputs
    IF packages is empty OR fleet is empty:
        RETURN empty list

    // Initialize
    routes = empty list
    available_vehicles = copy of fleet

    // YOUR ALGORITHM HERE
    // Example: Sweep Algorithm Pseudocode

    // 1. Convert package destinations to polar coordinates
    FOR each package IN packages:
        angle = atan2(package.y - depot.y, package.x - depot.x)
        package.polar_angle = angle

    // 2. Sort packages by polar angle
    sorted_packages = SORT packages BY polar_angle

    // 3. Sweep and create routes
    current_route = new Route with first vehicle
    FOR each package IN sorted_packages:
        IF current_route can fit package:
            ADD package to current_route
        ELSE:
            FINISH current_route
            START new route with next vehicle
            ADD package to new route

    // 4. Optimize each route (TSP)
    FOR each route IN routes:
        route.stops = optimize_order(route.packages)

    RETURN routes
```

---

## Phase 3: Implementation

### Step 3.1: Create Your Agent File

Create a new file: `src/agents/student_agent.py`

**Template to Start With:**

```python
"""
Student routing agent implementation.

[DESCRIBE YOUR ALGORITHM HERE]

Time Complexity: O(?)
Space Complexity: O(?)

Trade-offs:
+ [Advantages]
- [Disadvantages]
"""

from typing import List
from .base_agent import RouteAgent
from ..models import Package, Vehicle, Route, DeliveryMap


class StudentAgent(RouteAgent):
    """
    [YOUR ALGORITHM NAME] routing agent.

    Algorithm:
    1. [Step 1]
    2. [Step 2]
    3. [Step 3]
    """

    def __init__(self, delivery_map: DeliveryMap):
        """Initialize student agent."""
        super().__init__(delivery_map, "Student Agent")
        self.description = "Student implementation of [ALGORITHM NAME]"

    def plan_routes(self, packages: List[Package], fleet: List[Vehicle]) -> List[Route]:
        """
        Create routes using [YOUR ALGORITHM].

        Args:
            packages: List of packages to deliver
            fleet: Available vehicles

        Returns:
            List of routes
        """
        # Validate inputs
        if not self.validate_inputs(packages, fleet):
            return []

        print(f"[{self.name}] Planning routes for {len(packages)} packages...")

        # YOUR IMPLEMENTATION HERE
        routes = []

        # Example structure:
        # 1. Process packages (sort, cluster, etc.)
        # 2. Assign to vehicles (create Route objects)
        # 3. Optimize stop order for each route

        # TODO: Implement your algorithm

        # Print summary
        total_assigned = sum(len(r.packages) for r in routes)
        print(f"[{self.name}] Created {len(routes)} routes with {total_assigned} packages")

        return routes

    # Add helper methods as needed
    def _your_helper_method(self, param):
        """Helper method for [SPECIFIC TASK]."""
        pass
```

### Step 3.2: Use AI Assistants to Help Code

**Using ChatGPT/Claude (These have context of your codebase):**

**Prompt Examples:**
```
"I want to implement a Sweep Algorithm for the Delivery Fleet Game.
The algorithm should:
1. Convert package destinations to polar coordinates from depot
2. Sort by angle
3. Assign packages in sweep order to vehicles

Can you help me write the _assign_packages_by_sweep() method?
I need it to return a list of Route objects.
Here's the Route constructor: Route(vehicle, packages, stops, delivery_map)"
```

```
"I have a list of packages assigned to a route. I need to optimize
the delivery order to minimize distance. Can you write a method
that uses nearest neighbor TSP starting from depot (0, 0)?

The method should:
- Take List[Package] as input
- Return List[tuple] of ordered stops
- Use self.delivery_map.distance(loc1, loc2) for distances"
```

**Best Practices:**
- Share relevant code snippets with the AI
- Ask for explanations, not just code
- Request comments explaining the logic
- Test small pieces before integrating

### Step 3.3: Implement Key Components

**⚠️ Before You Start - Route Properties Cheat Sheet:**
```python
# When working with routes, remember:
route.total_volume              # ✅ Current load (sum of package volumes)
route.vehicle.vehicle_type.capacity_m3  # ✅ Max capacity
route.add_package(pkg)          # ✅ Returns True/False
route.packages                  # ✅ List of Package objects
route.stops                     # ✅ List of (x, y) tuples

# Common mistakes to avoid:
route.current_load              # ❌ Doesn't exist - use total_volume
route.capacity                  # ❌ Doesn't exist - use vehicle.vehicle_type.capacity_m3
```

**Component 1: Package Assignment Strategy**

Choose one approach:

**Option A: First-Fit Decreasing (Simple)**
```python
def _assign_packages_first_fit(self, packages: List[Package], fleet: List[Vehicle]) -> List[Route]:
    """Assign packages using first-fit decreasing."""
    # Sort by volume (largest first)
    sorted_packages = sorted(packages, key=lambda p: p.volume_m3, reverse=True)

    routes = []
    available_vehicles = fleet.copy()

    for pkg in sorted_packages:
        # Try existing routes
        placed = False
        for route in routes:
            if route.add_package(pkg):
                placed = True
                break

        # Need new vehicle
        if not placed and available_vehicles:
            vehicle = available_vehicles.pop(0)
            new_route = Route(vehicle, [pkg], [], self.delivery_map)
            routes.append(new_route)

    return routes
```

**Option B: Clustering by Location (Advanced)**
```python
import math

def _assign_packages_by_clusters(self, packages: List[Package], fleet: List[Vehicle]) -> List[Route]:
    """Assign packages by geographic clusters."""
    # Simple angle-based clustering
    clusters = [[] for _ in range(len(fleet))]

    for pkg in packages:
        # Calculate angle from depot
        dx = pkg.destination[0]
        dy = pkg.destination[1]
        angle = math.atan2(dy, dx)

        # Assign to cluster based on angle
        cluster_index = int((angle + math.pi) / (2 * math.pi) * len(fleet)) % len(fleet)
        clusters[cluster_index].append(pkg)

    # Create routes from clusters
    routes = []
    for i, cluster in enumerate(clusters):
        if cluster and i < len(fleet):
            # Create route, respecting capacity
            route = Route(fleet[i], [], [], self.delivery_map)
            for pkg in cluster:
                if not route.add_package(pkg):
                    print(f"Package {pkg.id} doesn't fit in cluster route")
            routes.append(route)

    return routes
```

**Component 2: Route Optimization (TSP)**

You can use the existing `Router` class:

```python
from ..core import Router

def _optimize_route_stops(self, packages: List[Package]) -> List[tuple]:
    """Optimize stop order using nearest neighbor TSP."""
    if not packages:
        return []

    destinations = [pkg.destination for pkg in packages]
    router = Router()

    optimized_stops = router.nearest_neighbor_tsp(
        destinations,
        self.delivery_map.depot,
        self.delivery_map
    )

    return optimized_stops
```

### Step 3.4: Complete Implementation Example

Here's a complete simple agent using sweep algorithm:

```python
"""
Student routing agent - Sweep Algorithm implementation.

The Sweep Algorithm uses polar angle sorting to create geographically
efficient routes. It sweeps around the depot like a radar, grouping
nearby packages together.

Time Complexity: O(n log n) for sorting + O(n²) for TSP = O(n²)
Space Complexity: O(n)

Trade-offs:
+ Good for radially distributed packages
+ Simple to understand and implement
+ Fast execution
- Not optimal for clustered distributions
- No value optimization
"""

import math
from typing import List
from .base_agent import RouteAgent
from ..models import Package, Vehicle, Route, DeliveryMap
from ..core import Router


class StudentAgent(RouteAgent):
    """
    Sweep Algorithm routing agent.

    Algorithm:
    1. Calculate polar angle for each package from depot
    2. Sort packages by angle (sweep clockwise)
    3. Assign packages to vehicles in sweep order (respecting capacity)
    4. Optimize each route using Nearest Neighbor TSP
    """

    def __init__(self, delivery_map: DeliveryMap):
        """Initialize student agent."""
        super().__init__(delivery_map, "Student Agent (Sweep)")
        self.description = "Sweep algorithm with polar angle sorting"
        self.router = Router()

    def plan_routes(self, packages: List[Package], fleet: List[Vehicle]) -> List[Route]:
        """
        Create routes using sweep algorithm.

        Args:
            packages: List of packages to deliver
            fleet: Available vehicles

        Returns:
            List of routes
        """
        if not self.validate_inputs(packages, fleet):
            return []

        print(f"[{self.name}] Planning routes for {len(packages)} packages...")

        # Step 1: Calculate polar angles and sort
        packages_with_angles = []
        for pkg in packages:
            dx = pkg.destination[0] - self.delivery_map.depot[0]
            dy = pkg.destination[1] - self.delivery_map.depot[1]
            angle = math.atan2(dy, dx)
            packages_with_angles.append((angle, pkg))

        # Sort by angle (sweep clockwise)
        packages_with_angles.sort(key=lambda x: x[0])
        sorted_packages = [pkg for angle, pkg in packages_with_angles]

        # Step 2: Assign packages in sweep order
        routes = []
        available_vehicles = fleet.copy()
        current_route = None

        for pkg in sorted_packages:
            # Try to add to current route
            if current_route and current_route.add_package(pkg):
                continue

            # Need new vehicle
            if available_vehicles:
                vehicle = available_vehicles.pop(0)
                current_route = Route(vehicle, [pkg], [], self.delivery_map)
                routes.append(current_route)
            else:
                print(f"[{self.name}] Warning: No more vehicles for package {pkg.id}")

        # Step 3: Optimize stop order for each route
        for route in routes:
            route.stops = self._optimize_route_stops(route.packages)

        # Summary
        total_assigned = sum(len(r.packages) for r in routes)
        print(f"[{self.name}] Created {len(routes)} routes with {total_assigned} packages")

        return routes

    def _optimize_route_stops(self, packages: List[Package]) -> List[tuple]:
        """Optimize stop order using nearest neighbor TSP."""
        if not packages:
            return []

        destinations = [pkg.destination for pkg in packages]
        optimized_stops = self.router.nearest_neighbor_tsp(
            destinations,
            self.delivery_map.depot,
            self.delivery_map
        )

        return optimized_stops
```

---

## Phase 4: Integration & Testing

### Step 4.1: Import Your Agent in main_pygame.py

Open `main_pygame.py` and add your import at the top (around line 15-20):

```python
# Find this section:
from src.agents import GreedyAgent, BacktrackingAgent, PruningBacktrackingAgent

# Add your agent:
from src.agents import GreedyAgent, BacktrackingAgent, PruningBacktrackingAgent, StudentAgent
```

### Step 4.2: Register Your Agent in the Engine

Find the `_register_agents()` method (around line 192) and add your agent:

```python
def _register_agents(self):
    """Register all routing agents."""
    self.engine.register_agent("greedy", GreedyAgent(self.engine.delivery_map))
    self.engine.register_agent("greedy_2opt", GreedyAgent(self.engine.delivery_map, use_2opt=True))
    self.engine.register_agent("backtracking", BacktrackingAgent(self.engine.delivery_map, max_packages=12))
    self.engine.register_agent("pruning_backtracking",
                               PruningBacktrackingAgent(self.engine.delivery_map, max_packages=15))

    # ADD YOUR AGENT HERE:
    self.engine.register_agent("student", StudentAgent(self.engine.delivery_map))
    # Optional: Add 2-opt variant
    self.engine.register_agent("student_2opt", StudentAgent(self.engine.delivery_map, use_2opt=True))
```

**Notes:**
- The first parameter ("student") is the ID used internally
- Choose a short, descriptive name
- Make sure to pass `self.engine.delivery_map` to your agent

### Step 4.3: Add Radio Buttons to the GUI

**IMPORTANT:** Registering the agent is not enough - you must also add it to the GUI radio buttons!

Find the agent radio buttons section (around line 250) in the `_create_ui_components()` method:

**Before (only 4 agents shown):**
```python
self.agent_radios = [
    RadioButton(radio_x, radio_y, "Greedy", "agent", "greedy"),
    RadioButton(radio_x, radio_y + 35, "Greedy+2opt", "agent", "greedy_2opt"),
    RadioButton(radio_x, radio_y + 70, "Backtrack", "agent", "backtracking"),
    RadioButton(radio_x, radio_y + 105, "Pruning BT", "agent", "pruning_backtracking"),
]
```

**After (with your agents added):**
```python
self.agent_radios = [
    RadioButton(radio_x, radio_y, "Greedy", "agent", "greedy"),
    RadioButton(radio_x, radio_y + 35, "Greedy+2opt", "agent", "greedy_2opt"),
    RadioButton(radio_x, radio_y + 70, "Backtrack", "agent", "backtracking"),
    RadioButton(radio_x, radio_y + 105, "Pruning BT", "agent", "pruning_backtracking"),
    RadioButton(radio_x, radio_y + 140, "Student", "agent", "student"),  # ADD THIS
    RadioButton(radio_x, radio_y + 175, "Student+2opt", "agent", "student_2opt"),  # ADD THIS
]
```

**RadioButton Parameters:**
- `radio_x, radio_y + offset`: Position (offset by 35px per button)
- `"Student"`: Display label shown in GUI
- `"agent"`: Radio group (keep as "agent")
- `"student"`: Value that matches your registered agent ID

### Step 4.4: Adjust Panel Sizes (if needed)

If you're adding many agents, you may need to expand the AGENTS panel. Find the panels section (around line 208):

**Before:**
```python
self.agent_panel = Panel(SIDEBAR_X + 10, SIDEBAR_START + 285, SIDEBAR_WIDTH - 20, 180, "AGENTS")
self.controls_panel = Panel(SIDEBAR_X + 10, SIDEBAR_START + 475, SIDEBAR_WIDTH - 20, 320, "CONTROLS")
```

**After (with more space for 6 agents):**
```python
self.agent_panel = Panel(SIDEBAR_X + 10, SIDEBAR_START + 285, SIDEBAR_WIDTH - 20, 250, "AGENTS")  # Increased height
self.controls_panel = Panel(SIDEBAR_X + 10, SIDEBAR_START + 545, SIDEBAR_WIDTH - 20, 320, "CONTROLS")  # Adjusted Y position
```

**Also update the control buttons Y position** (around line 262):
```python
btn_y = SIDEBAR_START + 585  # Adjusted for new panel position
```

**Notes:**
- Each radio button takes ~35px height
- Panel height = (number of agents × 35) + 40px padding
- Adjust `controls_panel` Y position to = `agent_panel Y + agent_panel height + 10`
- The third parameter in `RadioButton` is the display name, fourth is group, fifth is the ID

### Step 4.5: Test Syntax and Import Errors

Run the GUI to check for errors:

```bash
cd delivery_fleet_game
python main_pygame.py
```

**Common Errors:**

**ImportError: cannot import name 'StudentAgent'**
- Fix: Make sure `StudentAgent` is exported in `src/agents/__init__.py`

Add to `src/agents/__init__.py`:
```python
from .student_agent import StudentAgent

__all__ = ['RouteAgent', 'GreedyAgent', 'BacktrackingAgent',
           'PruningBacktrackingAgent', 'StudentAgent']
```

**TypeError: missing required positional argument**
- Fix: Check your `__init__` method signature matches base class

**AttributeError: 'StudentAgent' object has no attribute 'delivery_map'**
- Fix: Make sure you call `super().__init__(delivery_map, name)` in your `__init__`

### Step 4.6: Run Your First Test

Once the GUI launches successfully:

1. Click "Start Day" to begin Day 1
2. Select your agent from the radio buttons (should see "Student" option in the AGENTS panel)
3. Click "Plan Routes" button
4. Watch the console for debug output
5. Check if routes appear on the map

**What to Look For:**
- Do routes appear on the map?
- Are all packages assigned?
- Is the total distance reasonable?
- Any error messages in console?

### Step 4.7: Debugging Tips & Common Pitfalls

**Agent doesn't appear in GUI:**
- **Most common issue:** Did you add the RadioButton to `self.agent_radios` list? (Step 4.3)
- Verify the agent ID in RadioButton matches the one in `register_agent()`
- Check for typos in agent ID string
- Look for error messages during startup

**AttributeError: 'Route' object has no attribute 'current_load':**
- ✅ Use `route.total_volume` (sum of package volumes)
- ❌ NOT `route.current_load` (doesn't exist)
- Route properties are calculated automatically, don't set them manually

**AttributeError: 'Route' object has no attribute 'capacity':**
- ✅ Use `route.vehicle.vehicle_type.capacity_m3`
- ❌ NOT `route.capacity` (capacity is on the vehicle type, not route)

**No routes appear:**
- Check if `plan_routes()` returns a non-empty list
- Add print statements to see where code stops
- Verify `Route` objects are created correctly
- Make sure you're setting `route.stops` to a list of tuples

**Packages not assigned:**
- Check capacity constraints - are vehicles too small?
- Add logging to see which packages fail to fit
- Verify `route.add_package()` is working
- Remember: `add_package()` returns `True` if successful, `False` if capacity exceeded

**Poor performance (very long distances):**
- Check if `stops` are being set correctly
- Verify TSP optimization is being called
- Compare to greedy agent's distance
- Make sure stops include actual coordinates, not Package objects

**TypeError: 'NoneType' object is not iterable:**
- Your helper method might be returning `None` instead of an empty list
- Always return `[]` for empty cases, not `None`
- Check all return paths in your methods

**Routes look wrong on map:**
- Verify `stops` is a list of tuples: `[(x1, y1), (x2, y2), ...]`
- Don't include depot in stops - it's added automatically
- Make sure coordinates match `package.destination` values

---

## Phase 5: Iteration & Improvement

### Step 5.1: Benchmark Against Existing Agents

Run all agents on the same day and compare:

| Agent | Total Distance | Vehicles Used | Total Profit | Runtime |
|-------|----------------|---------------|--------------|---------|
| Greedy | 245.3 | 3 | $850 | <1s |
| Greedy 2-opt | 223.1 | 3 | $925 | <1s |
| Your Agent | ? | ? | ? | ? |

**How to Collect Data:**
1. Start Day 1
2. Run each agent and note the metrics shown in GUI
3. Record from the stats panel: Budget, Profit, etc.
4. Check console for distance and time information

### Step 5.2: Identify Weaknesses

**Questions to Ask:**
- Is my distance much longer than greedy? → Improve TSP optimization
- Do I use too many vehicles? → Better package packing
- Is my profit low? → Prioritize high-value packages
- Is it too slow? → Optimize algorithm complexity

**Common Issues:**

**Issue: High distance**
- Solution: Add 2-opt improvement after nearest neighbor
- Solution: Use better initial TSP heuristic (farthest insertion)

**Issue: Too many vehicles used**
- Solution: Sort packages by size before assignment
- Solution: Use bin-packing algorithm (First-Fit Decreasing)

**Issue: Low profit**
- Solution: Prioritize high-value packages first
- Solution: Consider value density (payment/volume)

**Issue: Slow execution**
- Solution: Reduce TSP iterations
- Solution: Use faster data structures (sets instead of lists)
- Solution: Avoid redundant distance calculations

### Step 5.3: Iterate on Your Design

**Improvement Ideas:**

**Level 1 - Quick Wins:**
1. Add 2-opt improvement to TSP
2. Sort packages by value before assignment
3. Use better vehicle selection (biggest first)

**Level 2 - Moderate Improvements:**
1. Implement hybrid approach (combine strategies)
2. Add local search after initial solution
3. Use clustering for better grouping

**Level 3 - Advanced:**
1. Implement Savings Algorithm
2. Add tabu search or simulated annealing
3. Use genetic algorithm for optimization

**Example Improvement: Adding Value Priority**

```python
# Before: Simple sweep by angle
sorted_packages = [pkg for angle, pkg in packages_with_angles]

# After: Sort by angle, but prioritize high-value packages
# within each angular sector
SECTOR_SIZE = math.pi / 4  # 45 degrees
sectors = {}
for angle, pkg in packages_with_angles:
    sector = int((angle + math.pi) / SECTOR_SIZE)
    if sector not in sectors:
        sectors[sector] = []
    sectors[sector].append(pkg)

# Sort each sector by value density
sorted_packages = []
for sector in sorted(sectors.keys()):
    sector_packages = sorted(sectors[sector],
                            key=lambda p: p.value_density,
                            reverse=True)
    sorted_packages.extend(sector_packages)
```

### Step 5.4: Test on Multiple Days

Don't just test on Day 1. Each day has different characteristics:

```bash
# In the GUI:
# 1. Complete Day 1 with your agent
# 2. Click "Next Day"
# 3. Start Day 2 and test again
# 4. Compare performance across days
```

**Why This Matters:**
- Day 1 might have easy package distributions
- Day 3+ might have more packages (stress test)
- Different distributions test algorithm robustness

### Step 5.5: Document Your Results

Create a `MY_AGENT_ANALYSIS.md` file:

```markdown
# My Agent Analysis

## Algorithm Description
[Describe your algorithm in 2-3 sentences]

## Implementation Details
- **Time Complexity:** O(?)
- **Space Complexity:** O(?)
- **Key Data Structures:** [Lists, dictionaries, etc.]

## Performance Results

### Day 1
- Total Distance: 230.5 km
- Vehicles Used: 3
- Profit: $890
- Runtime: 0.2s

### Day 2
- Total Distance: 245.8 km
- Vehicles Used: 4
- Profit: $1150
- Runtime: 0.3s

## Comparison to Greedy

| Metric | My Agent | Greedy | % Difference |
|--------|----------|--------|--------------|
| Distance | 230.5 | 245.3 | -6% (better) |
| Profit | $890 | $850 | +5% (better) |

## Strengths
- [List your agent's advantages]

## Weaknesses
- [List areas for improvement]

## Future Improvements
1. [Idea 1]
2. [Idea 2]
```

---

## Tips for Success

### General Best Practices

1. **Start Simple:** Get a basic working version first, then optimize
2. **Test Incrementally:** Test each component separately before integrating
3. **Use Print Statements:** Debug with strategic `print()` calls
4. **Read Error Messages:** They tell you exactly what's wrong
5. **Ask for Help:** Use AI assistants, but understand the code they give you

### Code Quality

1. **Add Comments:** Explain WHY, not just WHAT
2. **Use Descriptive Names:** `calculate_polar_angle()` not `calc_pa()`
3. **Keep Methods Short:** Each method should do ONE thing
4. **Handle Edge Cases:** What if no packages? No vehicles? Empty routes?

### Testing Strategy

1. **Unit Test:** Test helper methods individually
2. **Integration Test:** Test full agent in GUI
3. **Stress Test:** Try with many packages (Day 3+)
4. **Edge Case Test:** Try with 1 package, 1 vehicle, etc.

### Learning Resources

**Algorithms:**
- Wikipedia: "Vehicle Routing Problem"
- YouTube: Search "VRP visualization"
- Books: "Introduction to Algorithms" (CLRS)

**Python Help:**
- Python docs: docs.python.org
- Real Python: realpython.com
- Stack Overflow: stackoverflow.com

**AI Assistants:**
- ChatGPT: chat.openai.com
- Claude: claude.ai
- GitHub Copilot: In VS Code

---

## Common Student Questions

**Q: My agent is slower than greedy. Is that okay?**
A: Yes! If your solution quality is better, slight slowness is acceptable.
Aim for <5 seconds on Day 1.

**Q: Can I use external libraries like scipy or networkx?**
A: Generally yes, but check with instructor. Add to `requirements.txt` if needed.

**Q: What if I can't assign all packages?**
A: That's okay if capacity is insufficient. Make sure you:
1. Print a warning message
2. Assign as many as possible
3. Prioritize high-value packages

**Q: How do I know if my algorithm is "good"?**
A: Compare to greedy agent:
- Distance within 10% → Good
- Uses same or fewer vehicles → Good
- Higher profit → Excellent

**Q: Can I modify the base agent class?**
A: No, only inherit from it. Modify only files in `src/agents/` and registration in `main_pygame.py`.

**Q: What if my idea doesn't work well?**
A: That's part of learning! Document what you tried and why it didn't work.
Then try a different approach.

---

## Checklist for Completion

### Phase 1: Understanding
- [ ] Read and understand `base_agent.py`
- [ ] Study at least one existing agent implementation (e.g., `greedy_agent.py`)
- [ ] Understand Package, Vehicle, Route, and DeliveryMap models

### Phase 2: Design
- [ ] Research and choose an algorithm approach
- [ ] Write pseudocode on paper
- [ ] Understand time/space complexity of your approach

### Phase 3: Implementation
- [ ] Create `src/agents/student_agent.py` file
- [ ] Implement `__init__()` method (call `super().__init__()`)
- [ ] Implement `plan_routes()` method
- [ ] Add helper methods as needed
- [ ] Add docstrings and comments

### Phase 4: Integration
- [ ] Add agent to `src/agents/__init__.py` exports
- [ ] Import agent in `main_pygame.py` (top of file)
- [ ] Register agent in `_register_agents()` method
- [ ] **Add RadioButton to `self.agent_radios` list** (Step 4.3)
- [ ] Adjust panel sizes if needed (Step 4.4)
- [ ] Test GUI launches without errors
- [ ] Verify agent appears in AGENTS panel radio buttons

### Phase 5: Testing & Iteration
- [ ] Run agent and verify routes appear on map
- [ ] Check console output for debug messages
- [ ] Verify all packages are assigned (or unassigned reported)
- [ ] Benchmark against greedy agent (distance, profit, runtime)
- [ ] Test on multiple days (Day 1-5)
- [ ] Iterate and improve algorithm
- [ ] Document results and analysis

---

## Final Thoughts

Building a routing algorithm is challenging but rewarding. You'll learn about:
- Algorithm design and analysis
- Trade-offs between speed and quality
- Software engineering practices
- Problem-solving and debugging

Don't aim for perfection on the first try. Focus on:
1. Getting something working
2. Understanding how it works
3. Measuring performance
4. Iterating to improve

Good luck, and have fun experimenting!

---

## Quick Reference: Route API

Students often get confused about Route properties. Here's what you need to know:

### Creating a Route
```python
from ..models import Route

route = Route(
    vehicle=vehicle,           # Vehicle object from fleet
    packages=[pkg1, pkg2],     # List of Package objects
    stops=[],                  # Leave empty, fill after optimization
    delivery_map=self.delivery_map  # Pass from self
)
```

### Adding Packages to a Route
```python
# Try to add a package (checks capacity automatically)
if route.add_package(package):
    print("Package added successfully")
else:
    print("Package doesn't fit - capacity exceeded")
```

### Accessing Route Information
```python
# ✅ CORRECT - These properties exist and auto-calculate:
route.total_volume          # Sum of all package volumes (float)
route.total_distance        # Total km traveled (float)
route.total_cost            # Vehicle daily cost (float)
route.total_revenue         # Sum of package payments (float)
route.vehicle               # The Vehicle object
route.packages              # List of Package objects
route.stops                 # List of (x, y) tuples

# ✅ CORRECT - Accessing vehicle capacity:
route.vehicle.vehicle_type.capacity_m3

# ❌ WRONG - These don't exist:
route.current_load          # Use route.total_volume instead
route.capacity              # Use route.vehicle.vehicle_type.capacity_m3
route.load                  # Use route.total_volume instead
```

### Setting Route Stops
```python
# Extract destinations from packages
destinations = [pkg.destination for pkg in route.packages]

# Optimize order (using TSP, nearest neighbor, etc.)
optimized_stops = your_optimization_function(destinations)

# Set the stops (list of tuples)
route.stops = optimized_stops  # [(x1, y1), (x2, y2), ...]

# DON'T include depot in stops - it's added automatically
# DON'T use Package objects - use coordinate tuples
```

### Common Patterns
```python
# Check remaining capacity
remaining = route.vehicle.vehicle_type.capacity_m3 - route.total_volume

# Calculate capacity utilization
utilization = (route.total_volume / route.vehicle.vehicle_type.capacity_m3) * 100

# Get number of packages
num_packages = len(route.packages)

# Calculate profit for this route
profit = route.total_revenue - route.total_cost
```

---

## Quick Reference: Minimum Steps to Add an Agent

For experienced students, here's the absolute minimum to get your agent working:

### 1. Create Agent File
`src/agents/your_agent.py`:
```python
from .base_agent import RouteAgent
from ..models import Package, Vehicle, Route

class YourAgent(RouteAgent):
    def __init__(self, delivery_map):
        super().__init__(delivery_map, "Your Agent Name")

    def plan_routes(self, packages, fleet):
        if not self.validate_inputs(packages, fleet):
            return []
        # Your algorithm here
        return routes  # List[Route]
```

### 2. Export in `src/agents/__init__.py`
```python
from .your_agent import YourAgent
__all__ = [..., 'YourAgent']
```

### 3. Import in `main_pygame.py` (line ~20)
```python
from src.agents import ..., YourAgent
```

### 4. Register in `main_pygame.py` `_register_agents()` (line ~192)
```python
self.engine.register_agent("your_id", YourAgent(self.engine.delivery_map))
```

### 5. **ADD TO GUI** in `main_pygame.py` `_create_ui_components()` (line ~250)
```python
self.agent_radios = [
    # ... existing agents ...
    RadioButton(radio_x, radio_y + 140, "YourName", "agent", "your_id"),
]
```

### 6. Expand Panel (line ~211)
```python
self.agent_panel = Panel(SIDEBAR_X + 10, SIDEBAR_START + 285, SIDEBAR_WIDTH - 20, 250, "AGENTS")
self.controls_panel = Panel(SIDEBAR_X + 10, SIDEBAR_START + 545, ...)
# Update btn_y = SIDEBAR_START + 585
```

**Most Common Mistake:** Forgetting step 5 (adding RadioButton) - agent won't appear in GUI!

---

**Created for the Delivery Fleet Game**
**Version 1.1 - Updated with GUI Integration Steps**
