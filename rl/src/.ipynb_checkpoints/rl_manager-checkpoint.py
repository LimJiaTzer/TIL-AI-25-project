import numpy as np
import random
import heapq
from enum import IntEnum
import os
from collections import deque

# --- Imports for the Scout Agent ---
try:
    from dqn_agent import DQNAgent, MAX_STEPS_PER_EPISODE_DEFAULT as SCOUT_MAX_STEPS_DEFAULT
except ImportError as e:
    print(f"ERROR in RLManager: Failed to import DQNAgent from dqn_agent.py. Error: {e}")
    class DQNAgent:
        def select_action(self, obs): return 4 # Fallback STAY
    class QNetwork: pass
    SCOUT_MAX_STEPS_DEFAULT = 100

# --- Environment Action Definition ---
class EnvAction(IntEnum):
    FORWARD = 0; BACKWARD = 1; LEFT = 2; RIGHT = 3; STAY = 4

# --- Canonical Direction Definitions ---
CANONICAL_DIR_NORTH = 0; CANONICAL_DIR_EAST = 1; CANONICAL_DIR_SOUTH = 2; CANONICAL_DIR_WEST = 3
CANONICAL_DIR_OFFSETS = {
    CANONICAL_DIR_NORTH: np.array([0, -1]), CANONICAL_DIR_EAST:  np.array([1, 0]),
    CANONICAL_DIR_SOUTH: np.array([0, 1]),  CANONICAL_DIR_WEST:  np.array([-1, 0])
}
ENV_OBS_DIR_TO_CANONICAL_DIR = { 0: CANONICAL_DIR_EAST, 1: CANONICAL_DIR_SOUTH, 2: CANONICAL_DIR_WEST, 3: CANONICAL_DIR_NORTH }

# --- Viewcone Bit Indices & Environment Constants ---
BIT_IDX_SCOUT = 2; BIT_IDX_GUARD = 3
GRID_SIZE = 16; MAX_X = GRID_SIZE; MAX_Y = GRID_SIZE
ENV_VIEWCONE_PARAMS = (2, 2, 2, 4); VIEWCONE_LENGTH = 7; VIEWCONE_WIDTH = 5

# --- Wall Data (Ensure this is complete and correct) ---
ORIGINAL_FIXED_WALLS_SET = {
    ((-1, 0), (0, 0)), ((-1, 1), (0, 1)), ((-1, 2), (0, 2)), ((-1, 3), (0, 3)), ((-1, 4), (0, 4)), ((-1, 5), (0, 5)), ((-1, 6), (0, 6)), ((-1, 7), (0, 7)), ((-1, 8), (0, 8)), ((-1, 9), (0, 9)), ((-1, 10), (0, 10)), ((-1, 11), (0, 11)), ((-1, 12), (0, 12)), ((-1, 13), (0, 13)), ((-1, 14), (0, 14)), ((-1, 15), (0, 15)), ((0, -1), (0, 0)), ((0, 2), (0, 3)), ((0, 4), (0, 5)), ((0, 4), (1, 4)), ((0, 8), (1, 8)), ((0, 9), (1, 9)), ((0, 10), (0, 11)), ((0, 12), (0, 13)), ((0, 15), (0, 16)), ((1, -1), (1, 0)), ((1, 2), (1, 3)), ((1, 3), (2, 3)), ((1, 4), (2, 4)), ((1, 5), (1, 6)), ((1, 7), (1, 8)), ((1, 7), (2, 7)), ((1, 9), (1, 10)), ((1, 9), (2, 9)), ((1, 10), (2, 10)), ((1, 11), (1, 12)), ((1, 11), (2, 11)), ((1, 14), (1, 15)), ((1, 15), (1, 16)), ((2, -1), (2, 0)), ((2, 0), (3, 0)), ((2, 2), (3, 2)), ((2, 7), (2, 8)), ((2, 8), (3, 8)), ((2, 9), (3, 9)), ((2, 10), (2, 11)), ((2, 14), (2, 15)), ((2, 15), (2, 16)), ((3, -1), (3, 0)), ((3, 1), (3, 2)), ((3, 8), (3, 9)), ((3, 10), (3, 11)), ((3, 12), (3, 13)), ((3, 12), (4, 12)), ((3, 13), (3, 14)), ((3, 13), (4, 13)), ((3, 14), (4, 14)), ((3, 15), (3, 16)), ((4, -1), (4, 0)), ((4, 1), (5, 1)), ((4, 2), (4, 3)), ((4, 2), (5, 2)), ((4, 9), (4, 10)), ((4, 10), (4, 11)), ((4, 10), (5, 10)), ((4, 11), (5, 11)), ((4, 12), (4, 13)), ((4, 14), (5, 14)), ((4, 15), (4, 16)), ((4, 15), (5, 15)), ((5, -1), (5, 0)), ((5, 0), (6, 0)), ((5, 1), (5, 2)), ((5, 8), (5, 9)), ((5, 11), (6, 11)), ((5, 12), (6, 12)), ((5, 13), (5, 14)), ((5, 13), (6, 13)), ((5, 14), (6, 14)), ((5, 15), (5, 16)), ((6, -1), (6, 0)), ((6, 1), (7, 1)), ((6, 2), (6, 3)), ((6, 2), (7, 2)), ((6, 8), (6, 9)), ((6, 11), (7, 11)), ((6, 12), (7, 12)), ((6, 13), (7, 13)), ((6, 15), (6, 16)), ((7, -1), (7, 0)), ((7, 0), (7, 1)), ((7, 2), (7, 3)), ((7, 15), (7, 16)), ((8, -1), (8, 0)), ((8, 0), (8, 1)), ((8, 2), (8, 3)), ((8, 14), (8, 15)), ((8, 15), (8, 16)), ((9, -1), (9, 0)), ((9, 1), (9, 2)), ((9, 1), (10, 1)), ((9, 2), (9, 3)), ((9, 7), (9, 8)), ((9, 8), (9, 9)), ((9, 13), (9, 14)), ((9, 15), (9, 16)), ((9, 15), (10, 15)), ((10, -1), (10, 0)), ((10, 0), (10, 1)), ((10, 1), (11, 1)), ((10, 2), (10, 3)), ((10, 3), (11, 3)), ((10, 4), (11, 4)), ((10, 5), (11, 5)), ((10, 6), (11, 6)), ((10, 7), (11, 7)), ((10, 8), (10, 9)), ((10, 8), (11, 8)), ((10, 11), (11, 11)), ((10, 12), (11, 12)), ((10, 13), (10, 14)), ((10, 13), (11, 13)), ((10, 14), (11, 14)), ((10, 15), (10, 16)), ((11, -1), (11, 0)), ((11, 0), (12, 0)), ((11, 4), (12, 4)), ((11, 5), (12, 5)), ((11, 7), (12, 7)), ((11, 8), (12, 8)), ((11, 9), (12, 9)), ((11, 10), (12, 10)), ((11, 11), (12, 11)), ((11, 12), (11, 13)), ((11, 15), (11, 16)), ((11, 15), (12, 15)), ((12, -1), (12, 0)), ((12, 1), (12, 2)), ((12, 1), (13, 1)), ((12, 2), (12, 3)), ((12, 5), (13, 5)), ((12, 6), (12, 7)), ((12, 7), (13, 7)), ((12, 8), (13, 8)), ((12, 9), (13, 9)), ((12, 10), (13, 10)), ((12, 12), (13, 12)), ((12, 13), (12, 14)), ((12, 15), (12, 16)), ((13, -1), (13, 0)), ((13, 0), (13, 1)), ((13, 1), (13, 2)), ((13, 2), (13, 3)), ((13, 3), (14, 3)), ((13, 4), (13, 5)), ((13, 4), (14, 4)), ((13, 7), (14, 7)), ((13, 9), (14, 9)), ((13, 11), (14, 11)), ((13, 12), (14, 12)), ((13, 13), (14, 13)), ((13, 15), (13, 16)), ((14, -1), (14, 0)), ((14, 0), (14, 1)), ((14, 1), (14, 2)), ((14, 2), (14, 3)), ((14, 3), (15, 3)), ((14, 4), (14, 5)), ((14, 5), (14, 6)), ((14, 6), (14, 7)), ((14, 7), (14, 8)), ((14, 9), (14, 10)), ((14, 10), (14, 11)), ((14, 12), (15, 12)), ((14, 13), (15, 13)), ((14, 14), (15, 14)), ((14, 15), (14, 16)), ((14, 15), (15, 15)), ((15, -1), (15, 0)), ((15, 0), (16, 0)), ((15, 1), (16, 1)), ((15, 2), (16, 2)), ((15, 3), (16, 3)), ((15, 4), (16, 4)), ((15, 5), (15, 6)), ((15, 5), (16, 5)), ((15, 6), (16, 6)), ((15, 7), (15, 8)), ((15, 7), (16, 7)), ((15, 8), (15, 9)), ((15, 8), (16, 8)), ((15, 9), (16, 9)), ((15, 10), (16, 10)), ((15, 11), (16, 11)), ((15, 12), (16, 12)), ((15, 13), (16, 13)), ((15, 14), (16, 14)), ((15, 15), (15, 16)), ((15, 15), (16, 15)),
}
WALLS_FOR_GRAPH = set()
if ORIGINAL_FIXED_WALLS_SET:
    for wall_edge in ORIGINAL_FIXED_WALLS_SET: WALLS_FOR_GRAPH.add(tuple(sorted(wall_edge)))
else: print("RLManager WARNING: ORIGINAL_FIXED_WALLS_SET is empty.")

# --- Environment Helper Functions/Import ---
HELPERS_FROM_ENV = {}
try:
    from til_environment import helpers as env_helpers_module
    HELPERS_FROM_ENV = {'idx_to_view': env_helpers_module.idx_to_view, 'view_to_world': env_helpers_module.view_to_world}
except (ImportError, AttributeError) as e:
    print(f"RLManager WARNING: Could not import environment helpers: {e}. Viewcone parsing will be DISABLED.")

# --- Pathfinding & Map Utilities ---
def manhattan_dist_local(pos1, pos2): return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
def build_traversable_graph(max_x, max_y, walls):
    graph = {}
    for x in range(max_x):
        for y in range(max_y):
            node, graph[node] = (x, y), []
            for dx, dy in CANONICAL_DIR_OFFSETS.values():
                nx, ny = x + dx, y + dy
                if 0 <= nx < max_x and 0 <= ny < max_y and tuple(sorted((node, (nx, ny)))) not in walls:
                    graph[node].append((nx, ny))
    return graph
PRE_BUILT_TRAVERSABLE_GRAPH = build_traversable_graph(MAX_X, MAX_Y, WALLS_FOR_GRAPH)
def dijkstra_path(graph, start, end):
    if not graph or start not in graph or end not in graph: return []
    if start == end: return [start]
    distances, prevs = {node: float('inf') for node in graph}, {node: None for node in graph}
    distances[start], pq = 0, [(0, start)]
    while pq:
        dist, curr = heapq.heappop(pq)
        if dist > distances[curr]: continue
        if curr == end:
            path = []
            while curr is not None: path.append(curr); curr = prevs[curr]
            return path[::-1]
        for neighbor in graph.get(curr, []):
            new_dist = dist + 1
            if new_dist < distances[neighbor]:
                distances[neighbor], prevs[neighbor] = new_dist, curr
                heapq.heappush(pq, (new_dist, neighbor))
    return []

# --- Base Agent Class for Movement ---
class MovementAgentBase:
    def __init__(self, traversable_graph, viewcone_params, env_helpers):
        self.traversable_graph = traversable_graph
        self.viewcone_params, self.helpers = viewcone_params, env_helpers
        self.viewcone_length, self.viewcone_width = viewcone_params[2]+viewcone_params[3]+1, viewcone_params[0]+viewcone_params[1]+1
        self.debug = True

    def _get_turn_or_move_action(self, my_pos, my_dir, target_pos) -> EnvAction:
        if my_pos == target_pos: return EnvAction.STAY
        if manhattan_dist_local(my_pos, target_pos) > 1: return EnvAction.STAY
        target_dir_vec = (target_pos[0] - my_pos[0], target_pos[1] - my_pos[1])
        target_canonical_dir = -1
        for d, v in CANONICAL_DIR_OFFSETS.items():
            if v[0] == target_dir_vec[0] and v[1] == target_dir_vec[1]: target_canonical_dir = d; break
        if target_canonical_dir == -1: return EnvAction.STAY
        if my_dir == target_canonical_dir: return EnvAction.FORWARD
        if my_dir == (target_canonical_dir + 2) % 4: return EnvAction.BACKWARD
        diff = (target_canonical_dir - my_dir + 4) % 4
        return EnvAction.RIGHT if diff == 1 or diff == 2 else EnvAction.LEFT

# --- Heuristic Evasion Agent for Scout ---
class HeuristicEvasionAgent(MovementAgentBase):
    def get_action(self, my_pos, my_dir, guard_pos) -> int:
        best_spot, max_dist = my_pos, -1
        neighbors = self.traversable_graph.get(my_pos, [])
        if not neighbors: return EnvAction.STAY.value
        for neighbor in neighbors:
            path_from_neighbor = dijkstra_path(self.traversable_graph, neighbor, guard_pos)
            distance = len(path_from_neighbor) if path_from_neighbor else 99
            if distance > max_dist:
                max_dist, best_spot = distance, neighbor
        if self.debug: print(f"[EvasionAgent] Fleeing from {guard_pos}. Best spot to move to is {best_spot}.")
        return self._get_turn_or_move_action(my_pos, my_dir, best_spot).value

# --- Stateful Guard Agent ---
class StatefulGuardAgent(MovementAgentBase):
    SPAWN_AREAS = { 'BOTTOM_LEFT': ((0, 8), (4, 12)), 'BOTTOM_RIGHT': ((11, 9), (15, 13)), 'TOP_MIDDLE': ((7, 0), (11, 3)) }
    PATROL_ROUTES = {
        'BOTTOM_LEFT': [(4, 6), (0, 6), (8, 6), (8, 10), (4, 11), (0, 11), (0,5), (8, 5), (6, 7)],
        'BOTTOM_RIGHT': [(6, 6), (2, 2), (8, 8), (8, 11), (14, 10), (15, 15), (8, 11)],
        'TOP_MIDDLE': [(4, 4), (2, 2), (6, 6), (8, 8), (8, 11), (4, 11), (13, 5)]
    }
    def __init__(self, agent_id, traversable_graph, viewcone_params, env_helpers):
        super().__init__(traversable_graph, viewcone_params, env_helpers)
        self.agent_id = agent_id
        self.reset()

    def reset(self):
        self.state = 'INITIALIZING'
        self.patrol_route = []
        self.current_waypoint_index = 0
        self.last_known_scout_pos = None
        if self.debug: print(f"[{self.agent_id}] Guard State Reset.")

    def _get_scout_pos_from_view(self, obs):
        if not self.helpers: return None
        my_pos, my_dir = np.array(obs['location']), obs['direction']
        viewcone = np.array(obs['viewcone']).reshape((self.viewcone_length, self.viewcone_width))
        for r in range(self.viewcone_length):
            for c in range(self.viewcone_width):
                if (viewcone[r,c] >> BIT_IDX_SCOUT) & 1:
                    rel_coords = self.helpers['idx_to_view'](np.array([r,c]), self.viewcone_params)
                    return tuple(np.round(self.helpers['view_to_world'](my_pos, my_dir, rel_coords)).astype(int))
        return None

    def get_action(self, obs: dict) -> int:
        my_pos, my_dir = tuple(obs['location']), ENV_OBS_DIR_TO_CANONICAL_DIR[obs['direction']]
        scout_pos = self._get_scout_pos_from_view(obs)
        if scout_pos:
            if self.last_known_scout_pos != scout_pos: self.last_known_scout_pos = scout_pos
            if self.state != 'HUNTING': self.state = 'HUNTING'
        target = None
        if self.state == 'INITIALIZING':
            for area, (start, end) in self.SPAWN_AREAS.items():
                if start[0] <= my_pos[0] <= end[0] and start[1] <= my_pos[1] <= end[1]:
                    self.patrol_route = self.PATROL_ROUTES[area]; self.state = 'PATROLLING'; break
            if self.state == 'INITIALIZING': self.patrol_route, self.state = self.PATROL_ROUTES['BOTTOM_LEFT'], 'PATROLLING'
        if self.state == 'HUNTING':
            if self.last_known_scout_pos:
                target = self.last_known_scout_pos
                if my_pos == target and not scout_pos: self.state, self.last_known_scout_pos, target = 'PATROLLING', None, None
            else: self.state = 'PATROLLING'
        if self.state == 'PATROLLING':
            if self.current_waypoint_index >= len(self.patrol_route): self.state = 'ROAMING'
            else:
                target = self.patrol_route[self.current_waypoint_index]
                if my_pos == target: self.current_waypoint_index += 1; target = None
        if self.state == 'ROAMING':
            target = random.choice(self.traversable_graph.get(my_pos, [my_pos]))
        final_action = EnvAction.STAY
        if target and target != my_pos:
            path = dijkstra_path(self.traversable_graph, my_pos, target)
            if path and len(path) > 1: final_action = self._get_turn_or_move_action(my_pos, my_dir, path[1])
        elif self.state == 'ROAMING': final_action = random.choice([EnvAction.LEFT, EnvAction.RIGHT])
        return final_action.value

# --- RLManager Class with Heuristic Takeover ---
class RLManager:
    def __init__(self, train_mode: bool = False):
        self.scout_agent = None
        try:
            self.scout_agent = DQNAgent(cnn_input_channels=7, flat_input_size=8, n_actions=5, max_steps_per_episode_for_preprocessing=SCOUT_MAX_STEPS_DEFAULT)
            model_path = os.path.join('models', 'agent.pth')
            if os.path.exists(model_path):
                self.scout_agent.load_model_weights(model_path)
                print("RLManager: Successfully loaded Scout (DQNAgent) model.")
            else:
                print("RLManager WARNING: Scout model file not found. Scout will not function.")
                self.scout_agent = None
        except Exception as e:
            print(f"RLManager WARNING: Could not load/initialize Scout (DQNAgent) model. Error: {e}")
            self.scout_agent = None

        self.heuristic_evasion_agent = HeuristicEvasionAgent(traversable_graph=PRE_BUILT_TRAVERSABLE_GRAPH, viewcone_params=ENV_VIEWCONE_PARAMS, env_helpers=HELPERS_FROM_ENV)
        self.guard_agents = {}
        self.scout_pos_history = deque(maxlen=4)
        self.debug = True
        print("RLManager initialized with Heuristic Takeover logic.")

    def _find_visible_guard(self, obs: dict):
        if not HELPERS_FROM_ENV: return None
        my_pos, my_dir = np.array(obs['location']), obs['direction']
        viewcone = np.array(obs['viewcone']).reshape((VIEWCONE_LENGTH, VIEWCONE_WIDTH))
        for r in range(VIEWCONE_LENGTH):
            for c in range(VIEWCONE_WIDTH):
                if (viewcone[r,c] >> BIT_IDX_GUARD) & 1:
                    rel_coords = HELPERS_FROM_ENV['idx_to_view'](np.array([r,c]), ENV_VIEWCONE_PARAMS)
                    return tuple(np.round(HELPERS_FROM_ENV['view_to_world'](my_pos, my_dir, rel_coords)).astype(int))
        return None

    def _is_scout_stuck(self, my_pos):
        """Detects if the scout is stuck in an idle or repetitive loop."""
        # Condition 1: Stuck on the same tile (e.g., A -> A -> A)
        if len(self.scout_pos_history) >= 3 and len(set(self.scout_pos_history)) == 1:
            return True
        # Condition 2: Stuck in a 2-step back-and-forth loop (e.g., A -> B -> A -> B)
        if len(self.scout_pos_history) == 4:
            # History is [A, B, A, B]. Newest is at the end.
            if self.scout_pos_history[0] == self.scout_pos_history[2] and \
               self.scout_pos_history[1] == self.scout_pos_history[3] and \
               self.scout_pos_history[0] != self.scout_pos_history[1]:
                return True
        return False

    def _reset_scout_state(self):
        """Resets any state specific to the scout for a new episode."""
        self.scout_pos_history.clear()

    def rl(self, agent_id: str, observation: dict) -> int:
        if observation['step'] == 0:
            if observation.get('scout', 0) == 1:
                self._reset_scout_state()
            elif agent_id in self.guard_agents:
                self.guard_agents[agent_id].reset()

        # --- Scout Logic ---
        if observation.get('scout', 0) == 1:
            my_pos = tuple(observation['location'])
            my_dir = ENV_OBS_DIR_TO_CANONICAL_DIR[observation['direction']]
            self.scout_pos_history.append(my_pos)
            
            # --- Priority 1: Evasion from immediate threats ---
            guard_pos = self._find_visible_guard(observation)
            if guard_pos:
                path_from_guard = dijkstra_path(PRE_BUILT_TRAVERSABLE_GRAPH, guard_pos, my_pos)
                if path_from_guard and (len(path_from_guard) - 1) <= 3:
                    if self.debug: print(f"[{agent_id}] THREAT! Guard at {guard_pos} is too close. Heuristic Evasion Active.")
                    return self.heuristic_evasion_agent.get_action(my_pos, my_dir, guard_pos)
            
            # --- Priority 2: Anti-Stuck/Loop Heuristic ---
            if self._is_scout_stuck(my_pos):
                if self.debug: print(f"[{agent_id}] STUCK/LOOP DETECTED! Forcing a random move to break cycle.")
                neighbors = PRE_BUILT_TRAVERSABLE_GRAPH.get(my_pos, [])
                if neighbors:
                    # Make a random move to a new tile
                    return self.heuristic_evasion_agent._get_turn_or_move_action(my_pos, my_dir, random.choice(neighbors)).value
                else: # No neighbors, just turn
                    return random.choice([EnvAction.LEFT.value, EnvAction.RIGHT.value])

            # --- Default: Model Control ---
            if self.scout_agent:
                return self.scout_agent.select_action(observation)
            return EnvAction.STAY.value
        
        # --- Guard Logic ---
        else:
            if agent_id not in self.guard_agents:
                self.guard_agents[agent_id] = StatefulGuardAgent(agent_id=agent_id, traversable_graph=PRE_BUILT_TRAVERSABLE_GRAPH, viewcone_params=ENV_VIEWCONE_PARAMS, env_helpers=HELPERS_FROM_ENV)
            return self.guard_agents[agent_id].get_action(observation)