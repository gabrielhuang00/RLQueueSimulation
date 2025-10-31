from __future__ import annotations
import math
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from NetworkClasses import SchedulingPolicy, Network, Server, Queue, Station
import numpy as np
from collections import defaultdict


@dataclass
class MCTSNode:
    """Node in the MCTS tree representing a network state"""
    state: Dict[Tuple[str, str], int]  # (station_id, queue_id) -> queue_length
    action: Optional[Dict[Server, Queue]] = None  # Action that led to this state
    parent: Optional[MCTSNode] = None
    children: List[MCTSNode] = None
    visits: int = 0
    total_value: float = 0.0
    untried_actions: List[Dict[Server, Queue]] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.untried_actions is None:
            self.untried_actions = []
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    def add_child(self, state: Dict[Tuple[str, str], int], action: Dict[Server, Queue]) -> MCTSNode:
        child = MCTSNode(state=state, action=action, parent=self)
        self.children.append(child)
        return child


class MCTSPolicy(SchedulingPolicy):
    """Monte Carlo Tree Search policy for queue scheduling"""
    
    def __init__(self, 
                 num_rollouts: int = 100,
                 max_rollout_depth: int = 10,
                 exploration_param: float = math.sqrt(2),
                 rollout_policy: Optional[SchedulingPolicy] = None):
        self.num_rollouts = num_rollouts
        self.max_rollout_depth = max_rollout_depth
        self.exploration_param = exploration_param
        self.rollout_policy = rollout_policy  # Policy for rollouts
        
    def decide(self, net: Network, t: float, free_servers: Dict[str, List[Server]]) -> Dict[Server, Queue]:
        """Main MCTS search function"""
        if not free_servers:
            return {}
        
        # Create root node from current network state
        root_state = self._get_network_state(net)
        root = MCTSNode(state=root_state)
        
        # Generate all possible actions for root
        root.untried_actions = self._generate_actions(net, free_servers)
        
        # Run MCTS for specified number of rollouts
        for _ in range(self.num_rollouts):
            # Selection and expansion
            leaf = self._traverse(root, net, free_servers)
            
            # Simulation (rollout)
            value = self._rollout(leaf, net, free_servers)
            
            # Backpropagation
            self._backpropagate(leaf, value)
        
        # Return best action
        return self._best_child_action(root)
    
    def _traverse(self, node: MCTSNode, net: Network, free_servers: Dict[str, List[Server]]) -> MCTSNode:
        """Selection and expansion phase - traverse to leaf node"""
        current = node
        
        while True:
            # If node is not fully expanded, expand it
            if not current.is_fully_expanded():
                return self._expand(current, net, free_servers)
            
            # If fully expanded, select best child using UCB
            if not current.children:
                return current
            
            current = self._ucb_select(current)
        
        return current
    
    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        """UCB1 function for child selection"""
        best_score = float('-inf')
        best_child = None
        
        for child in node.children:
            if child.visits == 0:
                score = float('inf')
            else:
                exploitation = child.total_value / child.visits
                exploration = self.exploration_param * math.sqrt(math.log(node.visits) / child.visits)
                score = exploitation + exploration
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def _expand(self, node: MCTSNode, net: Network, free_servers: Dict[str, List[Server]]) -> MCTSNode:
        """Tree construction - expand node with new child"""
        if not node.untried_actions:
            return node
        
        # Select random untried action
        action = node.untried_actions.pop(random.randint(0, len(node.untried_actions) - 1))
        
        # Simulate applying this action to get new state
        new_state = self._simulate_action(node.state, action, net)
        
        # Create and add child node
        child = node.add_child(new_state, action)
        
        return child
    
    def _rollout(self, node: MCTSNode, net: Network, free_servers: Dict[str, List[Server]]) -> float:
        """Rollout function using rollout policy"""
        current_state = node.state.copy()
        
        for _ in range(self.max_rollout_depth):
            # Use rollout policy to select action
            action = self._rollout_policy(current_state, net, free_servers)
            
            if not action:
                break
            
            # Apply action and get new state
            current_state = self._simulate_action(current_state, action, net)
        
        # Calculate value of terminal state
        return self._value_function(current_state, net)
    
    def _rollout_policy(self, state: Dict[Tuple[str, str], int], 
                       net: Network, free_servers: Dict[str, List[Server]]) -> Dict[Server, Queue]:
        """Rollout policy function for action selection during simulation"""
        if self.rollout_policy:
            # Use provided rollout policy
            # Note: This would require creating a temporary network state
            # For now, use random policy
            pass
        
        # Default: random action selection
        return self._random_action(state, net, free_servers)
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagation - update values from leaf to root"""
        current = node
        
        while current is not None:
            current.visits += 1
            current.total_value += value
            current = current.parent
    
    def _value_function(self, state: Dict[Tuple[str, str], int], net: Network) -> float:
        """Value function for evaluating terminal states"""
        # Simple heuristic: negative of total queue length (prefer shorter queues)
        total_queue_length = sum(state.values())
        return -total_queue_length
    
    def _best_child_action(self, root: MCTSNode) -> Dict[Server, Queue]:
        """Select best child and return its action"""
        if not root.children:
            # Fallback to random action if no children
            return {}
        
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action if best_child.action else {}
    
    # Helper methods
    
    def _get_network_state(self, net: Network) -> Dict[Tuple[str, str], int]:
        """Extract current network state as queue lengths"""
        state = {}
        for station_id, station in net.stations.items():
            for queue_id, queue in station.queues.items():
                state[(station_id, queue_id)] = len(queue)
        return state
    
    def _generate_actions(self, net: Network, free_servers: Dict[str, List[Server]]) -> List[Dict[Server, Queue]]:
        """Generate all possible actions (server-queue assignments)"""
        actions = []
        
        # For simplicity, generate a subset of possible actions
        # In practice, you might want to limit this or use heuristics
        
        for station_id, servers in free_servers.items():
            station = net.stations[station_id]
            non_empty_queues = [q for q in station.queues.values() if len(q) > 0]
            
            if not non_empty_queues:
                continue
            
            # Generate actions for each server-queue combination
            for server in servers:
                for queue in non_empty_queues:
                    action = {server: queue}
                    actions.append(action)
        
        return actions
    
    def _simulate_action(self, state: Dict[Tuple[str, str], int], 
                        action: Dict[Server, Queue], net: Network) -> Dict[Tuple[str, str], int]:
        """Simulate applying action to state and return new state"""
        new_state = state.copy()
        
        # Decrease queue lengths for assigned queues
        for server, queue in action.items():
            key = (queue.station_id, queue.queue_id)
            if key in new_state and new_state[key] > 0:
                new_state[key] -= 1
        
        return new_state
    
    def _random_action(self, state: Dict[Tuple[str, str], int], 
                      net: Network, free_servers: Dict[str, List[Server]]) -> Dict[Server, Queue]:
        """Generate random action for rollout policy"""
        actions = self._generate_actions(net, free_servers)
        return random.choice(actions) if actions else {}