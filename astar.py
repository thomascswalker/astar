from typing import List
import time

try:
    from colorama import Fore  # type: ignore
except ImportError:
    Fore = None


class Vec2:
    x: int
    y: int

    def __init__(self, _x: int = 0, _y: int = 0) -> None:
        self.x = _x
        self.y = _y

    def __str__(self) -> str:
        return f"[{self.x}, {self.y}]"


class Node:
    index: int = -1
    pos: Vec2 = Vec2()
    parent: Vec2 = Vec2()
    g_cost: int = 0
    h_cost: int = 0
    is_walkable: bool = True
    in_path: bool = False

    @property
    def f_cost(self) -> int:
        return self.g_cost + self.h_cost

    def __str__(self) -> str:
        return f"{self.index} (Parent: {self.parent})"


class Grid:
    nodes: List[Node] = []
    size_x: int = 0
    size_y: int = 0

    def __init__(self, grid: str) -> None:
        # Split the input grid string by line (this is the Y size)
        lines = [line for line in grid.split("\n") if line]
        self.size_y = len(lines)

        # For each line (j = y pos, y = line)
        for j, y in enumerate(lines):
            # Split the line into elements separated by spaces (this is the X
            # size)
            chars = [e for e in y.strip().split(" ") if e]
            self.size_x = len(chars)

            # For each line element (i = x pos, y = char)
            for i, x in enumerate(chars):
                n = Node()  # Create a new node

                # If it's 'o' then it's walkable
                n.is_walkable = True if x == "o" else False
                n.pos = Vec2(i, j)
                n.index = (j * self.size_x) + i  # Determine 1D index

                # Add this to the list of nodes
                self.nodes.append(n)

    def is_pos_in_grid(self, pos: Vec2) -> bool:
        return 0 <= pos.x < self.size_x and 0 <= pos.y < self.size_y

    def get_node_from_index(self, index: Vec2) -> Node:
        pos = (index.y * self.size_x) + index.x
        if pos > len(self.nodes):
            print(pos)
            raise ValueError("Too high")
        return self.nodes[pos]

    def get_neighbors(self, source_node: Node) -> List[Node]:
        neighbors: List[Node] = []

        # Loop through [-1, -1] => [1, 1]
        for y in [-1, 0, 1]:
            for x in [-1, 0, 1]:
                # Compute the neighbor's X and Y
                cx = source_node.pos.x + x
                cy = source_node.pos.y + y

                # Check if we're inside the bounds of the grid
                if not self.is_pos_in_grid(Vec2(cx, cy)):
                    continue

                # If this is the root node, continue
                if x == 0 and y == 0:
                    continue

                if abs(x) == 1 and abs(y) == 1:
                    continue

                # Get the possible neighbor
                possible_neighbor = self.get_node_from_index(Vec2(cx, cy))

                # Skip neighbors which aren't walkable
                if not possible_neighbor.is_walkable:
                    continue

                # Append this neighbor to the list of neighbors
                neighbors.append(possible_neighbor)

        return neighbors

    def get_distance(self, node_a: Node, node_b: Node) -> int:
        ia = node_a.pos
        ib = node_b.pos

        # Get the difference between the two node's respective X and Y values
        dx = abs(ia.x - ib.x)
        dy = abs(ia.y - ib.y)

        # Get the delta between the min/max of dx:dy
        highest = max(dx, dy)
        lowest = min(dx, dy)
        delta = highest - lowest

        # Distance is calculated by taking into account sqrt of 2 (1.4) and
        # just multiplying by 10 to make it a clean integer.
        distance = (delta * 10) + (lowest * 14)
        return distance

    def retrace(self, start: Node, end: Node) -> List[Node]:
        print(f"Retracing from {end} to {start}")
        path: List[Node] = []

        # Start at the end node, working backwards
        current_node = end

        # While we have not reached the end...
        while current_node != start:
            # Add this node to the path
            path.append(current_node)

            # Set the current node to the parent of this node
            current_node = self.get_node_from_index(current_node.parent)

        # Reverse the path so we are at the start
        reversed_path = list(reversed(path))

        return reversed_path

    def find_path(self, start: Node, end: Node) -> List[Node]:
        # Create open and closed set, start open set with the start node
        open_set: List[Node] = [start]
        closed_set: List[Node] = []

        # While the open set is not empty...
        while len(open_set) != 0:
            # Set the current node to the cheapest node (by f/h scores)
            current_node = open_set[0]
            for node in open_set:
                f_compare = node.f_cost <= current_node.f_cost
                h_compare = node.h_cost < current_node.h_cost
                if f_compare and h_compare:
                    current_node = node

            # Remove this node from the open set and add it to the closed set
            open_set.remove(current_node)
            closed_set.append(current_node)

            # If the current node is the end node, we can trace back from the
            # end to construct the path
            if current_node == end:
                return self.retrace(start, end)

            # For each neighbor of the current node
            for neighbor in self.get_neighbors(current_node):
                in_open = neighbor in open_set
                in_closed = neighbor in closed_set

                # If we've already visited this neighbor and closed it, we'll
                # skip it
                if in_closed:
                    continue

                # Calculate distance from current to neighbor
                distance = self.get_distance(current_node, neighbor)
                # Calculate new cost of the neighbor
                new_cost = current_node.g_cost + distance

                # If this new cost is lower than the current cost, and the node
                # is not in the open set
                if new_cost < neighbor.g_cost or not in_open:
                    # Set new cost
                    neighbor.g_cost = new_cost
                    # Set h cost to be distance from this neighbor to the end
                    neighbor.h_cost = self.get_distance(neighbor, end)
                    # Set the parent to be the current node
                    neighbor.parent = current_node.pos
                    # If the neighbor is not in the open set, add it
                    if not in_open:
                        open_set.append(neighbor)

        print("Path could not be found.")
        return []

    def set_path(self, path: List[Node]) -> None:
        for node in path:
            node.in_path = True

    def print(self) -> str:
        formatted_string = ""
        for i, n in enumerate(self.nodes):
            if not i % self.size_x:
                formatted_string += "\n"
            if n.in_path:
                formatted_string += Fore.GREEN if Fore else ""
                formatted_string += " O "
                continue
            if not n.is_walkable:
                formatted_string += Fore.RED if Fore else ""
                formatted_string += "|||"
                continue
            formatted_string += Fore.WHITE if Fore else ""
            formatted_string += "   "
        formatted_string += Fore.WHITE if Fore else ""
        return formatted_string

    def __str__(self) -> str:
        return self.print()


grid_string_1 = """o x x o o o o o o x x o o o o o
o o o o x o x o o o o o x o x o
x x x o o o x o x x x o o o x o
x x x x x x o o x x x x x x o o
x o o o x x o x x o o o x x o x
o o x o o x o o o o x o o o o x
x o o o x x o x x o o o x x o x
o x x o o o o o x x x o o o o o
o o o o x x x o o o o o x o x o
x x x x o o x o x x x o o o x o
x x x x x x o o x x x x x x o o
x o o o x x o x x o o o x x o x
o o x o o o o x o o x o o o o x
o x x x x x x o o x x x x x x o
o o o o o o x o o o o o o o o o
"""

grid_string_2 = """o o o o o o o
x x o o o o x
o o o o o o o
o x x x x x o
o o o o o o o
x x o o o o x
o o o o o o o
x x o o o o x
o o o o o o o
o x x x x x o
o o o o o o o
x x o o o o x
o o o o o o o
x x o o o o x
o o o o o o o
o x x x x x o
o o o o o o o
x x o o o o o
"""


start = time.perf_counter()
grid = Grid(grid_string_1)
path = grid.find_path(grid.nodes[0], grid.nodes[-1])
grid.set_path(path)
end = time.perf_counter()
print(f"Grid with {len(grid.nodes)} nodes:")
print(grid)
print(f"Solved in {round(end - start, 3)}s")
