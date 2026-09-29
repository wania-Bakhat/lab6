"""
searchAlgos.py
Graph, heuristic, GBFS and A* for the Emergency Supply Robot (hospital) problem.
These are the same algorithms as Task 2 / Task 3 in the notebook, wrapped so
that app.py can import and use them.
"""

import math
import heapq

# ------------------------------------------------------------------
# Graph, Use Case: Emergency Supply Robot
# ------------------------------------------------------------------

locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6),
}

hospital_graph = {
    "Pharmacy": {
        "Main_Corridor": 2.2,
        "Patient_Wing": 4.1,
    },
    "Main_Corridor": {
        "Nursing_Station": 2.2,
    },
    "Patient_Wing": {
        "Laboratory": 5.0,
    },
    "Nursing_Station": {
        "Laboratory": 3.2,
        "Emergency_Ward": 6.0,
    },
    "Laboratory": {
        "Emergency_Ward": 3.2,
    },
    "Emergency_Ward": {},
}


# ------------------------------------------------------------------
# Heuristic: straight-line (Euclidean) distance from current to goal
# ------------------------------------------------------------------
def heuristic(current, goal):
    x1, y1 = locations[current]
    x2, y2 = locations[goal]
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ------------------------------------------------------------------
# Path reconstruction
# ------------------------------------------------------------------
def reconstruct_path(came_from, current):
    path = []
    while current is not None:
        path.append(current)
        current = came_from.get(current)
    path.reverse()
    return path


def path_cost(path, graph=hospital_graph):
    """Real cost of a path = sum of the edge weights along it."""
    return sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))


# ------------------------------------------------------------------
# Greedy Best-First Search   (priority = h(n))
# returns: path, total_cost, expansion_order   (path is None if no route)
# ------------------------------------------------------------------
def gbfs(start, goal, graph=hospital_graph):
    frontier = [(heuristic(start, goal), start)]
    came_from = {start: None}
    explored = set()
    expansion_order = []

    while frontier:
        _, current = heapq.heappop(frontier)

        if current in explored:
            continue
        explored.add(current)
        expansion_order.append(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, path_cost(path, graph), expansion_order

        for neighbor in graph[current]:
            if neighbor not in explored and neighbor not in came_from:
                came_from[neighbor] = current
                heapq.heappush(frontier, (heuristic(neighbor, goal), neighbor))

    return None, None, expansion_order


# ------------------------------------------------------------------
# A* Search   (priority = f(n) = g(n) + h(n))
# returns: path, total_cost, expansion_order   (path is None if no route)
# ------------------------------------------------------------------
def a_star(start, goal, graph=hospital_graph):
    g_cost = {start: 0}
    came_from = {start: None}
    frontier = [(heuristic(start, goal), start)]
    explored = set()
    expansion_order = []

    while frontier:
        _, current = heapq.heappop(frontier)

        if current in explored:          # stale queue entry, skip it
            continue
        explored.add(current)
        expansion_order.append(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, g_cost[goal], expansion_order

        for neighbor, weight in graph[current].items():
            tentative_g = g_cost[current] + weight
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                came_from[neighbor] = current
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(frontier, (f, neighbor))

    return None, None, expansion_order
