#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
import heapq
import torch.multiprocessing as mp
from collections import deque, OrderedDict
from typing import List, Dict, Tuple, Any, Optional
from mcts_NetworkClasses import (
    Network, 
    ExtendedSixClassNetwork,
    SchedulingPolicy,
    Server,
    Queue,
    EventType,
    LBFSPolicy,
    FIFONetPolicy
)

# --- Hyperparameters / Config ---
CONFIG = {
    # --- Network & NN API ---
    "L": 2,                       # L=2 for ExtendedSixClassNetwork
    "MAX_QUEUES_STATE": 10,       # Pad state vector to this size
    "MAX_STATIONS": 2,            # Max stations (for building action list)
    "MAX_QUEUES_PER_STATION": 3,  # Max queues per station (for action list)
    
    # --- NN & Training ---
    "learning_rate": 0.001,
    "buffer_size": 200000,
    "batch_size": 256, #Experiment with
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    # --- MCTS ---
    "mcts_simulations": 100,
    "c_puct": 1.5,
    "mcts_sim_horizon_s": 100.0,
    "temperature": 1.0,
    "dirichlet_alpha": 0.3,   # Alpha parameter for the noise
    "dirichlet_epsilon": 0.25, # Weight of noise (25%) vs. policy (75%)
    
    # --- Training Loop ---
    "num_train_loops": 100,
    "episodes_per_loop": 50,      # Generate 50 sim runs
    "train_steps_per_loop": 200,  # ML Epochs
    "sim_run_duration": 10000.0,    # Run each "real" sim for Xs
    "CATASTROPHE_SOJOURN_TIME": 80.0, # The "worst" score for normalization
    "seed": 1
}

# --- 1. The Neural Network ---

class AlphaZeroNN(nn.Module):
    """The 'Body-Head' neural network."""
    def __init__(self, state_size, action_space_size):
        super(AlphaZeroNN, self).__init__()
        
        self.state_size = state_size
        self.action_space_size = action_space_size
        
        # Shared "Body"
        self.body = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        
        # Policy Head (Predicts MCTS visit counts)
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space_size)
        )
        
        # Value Head (Predicts game outcome)
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # Squashes value to [-1, 1]
        )

    def forward(self, state):
        body_out = self.body(state)
        policy_logits = self.policy_head(body_out)
        value = self.value_head(body_out)
        return policy_logits, value

# --- 2. The Replay Buffer ---

class ReplayBuffer:
    """Stores (state, policy_target, value_target) tuples."""
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, policy_target, value_target):
        self.buffer.append((state, policy_target, value_target))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, policy_targets, value_targets = zip(*batch)
        
        return (np.array(states), 
                np.array(policy_targets), 
                np.array(value_targets).reshape(-1, 1))

    def __len__(self):
        return len(self.buffer)

# --- 3. The MCTS Node ---

class MCTSNode:
    def __init__(self, parent: Optional[MCTSNode] = None, prior_p: float = 0.0):
        self.parent = parent
        self.children: Dict[int, MCTSNode] = {}  # action_idx -> Node
        
        self.N = 0      # Visit count
        self.W = 0.0    # Total action value (sum of rewards)
        self.Q = 0.0    # Mean action value (W / N)
        self.P = prior_p  # Prior probability from NN
        
        self._state_vector: Optional[np.ndarray] = None
        self._is_terminal: bool = False

    def get_state_vector(self, sim_net: Network, max_queues: int) -> np.ndarray:
        """Gets the fixed-length state vector for the NN."""
        if self._state_vector is None:
            q_lengths_dict = sim_net.total_queue_lengths()
            # Sort by key: (S1, Q1), (S1, Q2), (S2, Q4)...
            ordered_q_lengths = OrderedDict(sorted(q_lengths_dict.items()))
            
            state = [length for length in ordered_q_lengths.values()]
            
            # Pad with zeros
            padding = [0] * (max_queues - len(state))
            self._state_vector = np.array(state + padding)
            
            if len(self._state_vector) > max_queues:
                self._state_vector = self._state_vector[:max_queues]
                
        return self._state_vector

    def expand(self, policy_probs: np.ndarray):
        """Expand this node using NN policy priors."""
        for action_idx, prob in enumerate(policy_probs):
            if prob > 0: # Only create nodes for legal actions
                if action_idx not in self.children:
                    self.children[action_idx] = MCTSNode(parent=self, prior_p=prob)

    def select_child_puct(self, c_puct: float) -> Tuple[int, MCTSNode]:
        """Select the action/child that maximizes the PUCT score."""
        best_score = -np.inf
        best_action_idx = -1
        best_child = None

        sqrt_self_N = math.sqrt(self.N)
        
        for action_idx, child in self.children.items():
            score = child.Q + c_puct * child.P * (sqrt_self_N / (1 + child.N))
            if score > best_score:
                best_score = score
                best_action_idx = action_idx
                best_child = child
        
        return best_action_idx, best_child

    def backpropagate(self, reward: float):
        """Update N and W values up the tree."""
        node = self
        while node is not None:
            node.N += 1
            node.W += reward
            node.Q = node.W / node.N
            node = node.parent

# --- 4. The MCTS Policy (The "Planner") ---

class MCTS_Policy(SchedulingPolicy):
    """
    This class is the MCTS planner. It plugs into the Network
    as its "policy" object.
    """
    def __init__(self, model: AlphaZeroNN, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = config["device"]
        
        self.master_action_list: List[Tuple[str, str]] = self._build_master_action_list()
        self.action_to_idx: Dict[Tuple[str, str], int] = {
            action: i for i, action in enumerate(self.master_action_list)
        }
        self.idx_to_action: Dict[int, Tuple[str, str]] = {
            i: action for i, action in enumerate(self.master_action_list)
        }
        self.NULL_ACTION_IDX = self.action_to_idx[("NULL", "NULL")]
        
        # for i, action in enumerate(self.master_action_list):
        #     print(f"  Idx {i}: {action}")

    def _build_master_action_list(self) -> List[Tuple[str, str]]:
        actions = []
        # Build based on L=2 network topology
        for i in range(1, self.config["L"] + 1):
            station_id = f"S{i}"
            server_id = f"{station_id}-s0" # Assumes 1 server 's0'
            for k in range(1, 4):
                class_id = 3 * (i - 1) + k
                queue_id = f"Q{class_id}"
                actions.append((server_id, queue_id))
                
        actions.append(("NULL", "NULL"))
        return actions

    def _get_action_mask(self, sim_net: Network, free_servers: Dict[str, List[Server]]) -> np.ndarray:
        mask = np.zeros(len(self.master_action_list), dtype=int)
        
        non_empty_queues = {
            (sid, qid) for sid, st in sim_net.stations.items()
            for qid, q in st.queues.items() if len(q) > 0
        }
        
        found_legal_action = False
        for station_id, srvs in free_servers.items():
            for srv in srvs:
                for action_idx, (srv_id, qid) in enumerate(self.master_action_list):
                    if srv.server_id == srv_id:
                        if (station_id, qid) in non_empty_queues:
                            mask[action_idx] = 1
                            found_legal_action = True
                            
        if not found_legal_action and free_servers:
            # If servers are free but no actions are possible,
            # the *only* legal action is to do nothing.
            mask[self.NULL_ACTION_IDX] = 1
            
        return mask

    def decide(
        self,
        net: Network,
        t: float,
        free_servers: Dict[str, List[Server]],
    ) -> Tuple[Dict[Server, Queue], MCTSNode, np.ndarray]:
        """
        Runs the full MCTS search and returns the best action,
        the root node (for training data), and the state vector.
        """
        
        # --- 1. Sequential Assignment ---
        # We only assign ONE server at a time.
        # We find the *first* free server and make a decision for it.
        # The Network will call us again if others are still free.
        
        server_to_assign: Optional[Server] = None
        station_id_for_server: Optional[str] = None 
        for st_id, srvs in free_servers.items():
            if srvs:
                server_to_assign = srvs[0]
                station_id_for_server = st_id     
                break
        
        if server_to_assign is None:
            # No free servers, return empty info
            root = MCTSNode()
            state_vec = root.get_state_vector(net, self.config["MAX_QUEUES_STATE"])
            return {}, root, state_vec
        
        search_free_servers = {station_id_for_server: [server_to_assign]}

        # --- 2. Run MCTS Search ---
        try:
            real_snapshot = net.clone()
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to clone network: {e}")
            root = MCTSNode()
            state_vec = root.get_state_vector(net, self.config["MAX_QUEUES_STATE"])
            return {}, root, state_vec # Fallback: do nothing

        root = MCTSNode()
        # Get the state vector for this root *before* search
        state_vec = root.get_state_vector(net, self.config["MAX_QUEUES_STATE"])
        
        for _ in range(self.config["mcts_simulations"]):
            node = root
            sim_net = real_snapshot.clone()
            sim_t = t
            sim_free_servers = search_free_servers
            
            # --- SELECTION ---
            path = [node]
            while node.children: 
                action_idx, node = node.select_child_puct(self.config["c_puct"])
                path.append(node)
                
                (sim_net, sim_t, sim_free_servers, is_terminal) = self._run_sim_step(
                    sim_net, 
                    self.idx_to_action[action_idx]
                )
                
                if is_terminal:
                    node._is_terminal = True
                    break
            
            # --- EXPANSION & EVALUATION ---
            value = 0.0 # Default terminal value
            if not node._is_terminal:
                leaf_state_vec = node.get_state_vector(sim_net, self.config["MAX_QUEUES_STATE"])
                state_tensor = torch.tensor(leaf_state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    policy_logits, value_tensor = self.model(state_tensor)
                
                value = value_tensor.item()
                action_mask = self._get_action_mask(sim_net, sim_free_servers)
                
                policy_logits = policy_logits.squeeze(0)
                policy_logits[action_mask == 0] = -torch.inf
                policy_probs = F.softmax(policy_logits, dim=0).cpu().numpy()
                
                node.expand(policy_probs)
            
            # --- BACKPROPAGATION ---
            node.backpropagate(value)

        # --- 3. Make Final Decision ---
        if not root.children:
            best_action_idx = self.NULL_ACTION_IDX
        else:
            # Select action based on visit counts
            best_action_idx = max(
                root.children, 
                key=lambda action_idx: root.children[action_idx].N
            )
        
        best_action: Tuple[str, str] = self.idx_to_action[best_action_idx]
        
        # --- 4. Translate Action to Assignment ---
        assignments: Dict[Server, Queue] = {}
        (srv_id, qid) = best_action
        
        if srv_id != "NULL":
            # Find the actual Server object
            srv = server_to_assign
            if srv.server_id == srv_id:
                # <-- FIX 4: Use the stored station ID to find the queue
                q = net.stations[station_id_for_server].queues[qid] 
                assignments[srv] = q
            else:
                # This is a logic error
                print(f"Warning: MCTS chose action for wrong server! {srv_id}")

        return assignments, root, state_vec

    def _run_sim_step(
        self, 
        sim_net: Network, 
        action: Tuple[str, str]
    ) -> Tuple[Network, float, Dict[str, List[Server]], bool]:
        """Applies one MCTS action and runs sim to next decision."""
        
        action_policy = FixedActionPolicy(action)
        sim_net.policy = action_policy
        
        free_now = sim_net._get_free_servers()
        if free_now:
            assignments = action_policy.decide(sim_net, sim_net.t, free_now)
            for srv, q in assignments.items():
                if len(q) == 0 or srv.busy: continue
                job = q.pop()
                dep_time = srv.start_service(job, sim_net.t)
                st_id = q.station_id
                st = sim_net.stations[st_id]
                server_idx = st.servers.index(srv)
                sim_net.schedule(dep_time, EventType.DEPARTURE, {"station_id": st_id, "server_idx": server_idx})
        
        t_start = sim_net.t
        (new_t, new_free_servers) = sim_net.run_until_next_decision()
        
        is_terminal = False
        if not sim_net._event_q: # Check if event queue is empty
            is_terminal = True
        if new_t > t_start + self.config["mcts_sim_horizon_s"]:
            is_terminal = True
            
        return sim_net, new_t, new_free_servers, is_terminal
    
    ### --- MODIFIED (Made Public) --- ###
    def get_policy_target(self, root: MCTSNode, temperature: float) -> np.ndarray:
        """Get the policy target (visit counts) to train the NN."""
        policy_target = np.zeros(len(self.master_action_list))
        if not root.children:
            policy_target[self.NULL_ACTION_IDX] = 1.0
            return policy_target
            
        visit_counts = np.array([
            child.N for child in root.children.values()
        ])
        action_indices = np.array([
            action_idx for action_idx in root.children.keys()
        ])
        
        if temperature == 0:
            best_action_local_idx = np.argmax(visit_counts)
            best_action_global_idx = action_indices[best_action_local_idx]
            policy_target[best_action_global_idx] = 1.0
        else:
            visit_counts_temp = visit_counts ** (1.0 / temperature)
            probs_sum = np.sum(visit_counts_temp)
            if probs_sum == 0:
                # All counts are 0, use uniform
                for idx in action_indices:
                    policy_target[idx] = 1.0 / len(action_indices)
            else:
                probs = visit_counts_temp / probs_sum
                for idx, prob in zip(action_indices, probs):
                    policy_target[idx] = prob
                
        return policy_target

# --- 5. Helper Policy for MCTS ---

class FixedActionPolicy(SchedulingPolicy):
    """A dummy policy that executes one pre-selected action."""
    def __init__(self, action: Tuple[str, str]):
        self.srv_id, self.qid = action

    def decide(
        self, 
        net: Network, 
        t: float, 
        free_servers: Dict[str, List[Server]]
    ) -> Dict[Server, Queue]:
        
        assignments = {}
        if self.srv_id == "NULL":
            return assignments
            
        for st_id, srvs in free_servers.items():
            for srv in srvs:
                if srv.server_id == self.srv_id:
                    q = net.stations[st_id].queues[self.qid]
                    assignments[srv] = q
                    return assignments
        return assignments

class Trainer:
    """The orchestrator that manages training."""
    def __init__(self, config: Dict[str, Any], model = None, start_states=None):
        self.config = config
        self.device = config["device"]
        self.seed = config["seed"]
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        temp_policy = MCTS_Policy(None, self.config) # Dummy to get sizes
        self.action_space_size = len(temp_policy.master_action_list)
        self.state_size = self.config["MAX_QUEUES_STATE"]
        self.start_states = start_states
        
        print(f"Trainer init: state_size={self.state_size}, action_space_size={self.action_space_size}")

        if model == None:
            self.model = AlphaZeroNN(
                self.state_size, 
                self.action_space_size
            ).to(self.device)
            
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=config["learning_rate"]
            )
        else: 
            self.model = model
        self.replay_buffer = ReplayBuffer(config["buffer_size"])
        
        self.mcts_policy = MCTS_Policy(self.model, self.config)
        self.mcts_policy.model.eval()

    def run_simulation_episode(self) -> float:
        """
        Simulates one full episode by running its *own* event loop
        to capture (state, policy) data at each decision.
        """
        if self.start_states:
            random_snapshot = random.choice(self.start_states)
            net = random_snapshot.clone()
            net.policy = self.mcts_policy
            
            # --- CRITICAL FIX: RESET STATS ---
            # We want to measure performance ONLY for this 500s window.
            # We must wipe the "LBFS History" from the metrics variables.
            
            # 1. Reset Time to 0 (so we measure duration relative to now)
            net.t = 0.0 
            
            # 2. Reset Areas to 0
            for st in net.stations.values():
                st._ql_area = {qid: 0.0 for qid in st.queues}
                st._sl_area = 0.0
            
            # 3. Reset Counters
            net.completed_jobs = 0
            net.sum_sojourn = 0.0
            # ---------------------------------
            
        else:
        # 1. Create a new "real" simulation
            net = ExtendedSixClassNetwork(
                policy=self.mcts_policy, # The MCTS policy
                L=self.config["L"],
                seed=random.randint(0, 1_000_000)
            )
            
        episode_history: List[Tuple[np.ndarray, np.ndarray]] = []
        
        # 2. Seed the simulation (from Network.run())
        if not net._seeded:
            for ap in net.arrivals:
                t_next = ap.schedule_next(net.t)
                net.schedule(t_next, EventType.ARRIVAL, {"ap": ap})
            net._seeded = True
        
        # 3. Run the "Outer" simulation event loop
        while net._event_q:
            if net._event_q[0].time > self.config["sim_run_duration"]:
                break
    
            ev = heapq.heappop(net._event_q)
    
            # --- This logic is copied from Network.run() ---
            dt = ev.time - net.t
            if dt > 0:
                for st in net.stations.values():
                    for qid, q in st.queues.items():
                        st._ql_area[qid] += len(q) * dt
                    num_busy_servers = sum(1 for srv in st.servers if srv.busy)
                    st._sl_area += num_busy_servers * dt
            net.t = ev.time
            # --- End copied logic ---
    
            # Handle event
            if ev.type == EventType.ARRIVAL:
                net._on_arrival(ev.payload["ap"])
            elif ev.type == EventType.DEPARTURE:
                net._on_departure(ev.payload["station_id"], ev.payload["server_idx"])
    
            # --- DATA CAPTURE HOOK ---
            # This is the "Scheduling decision" part
            free_servers = net._get_free_servers()
            
            while free_servers:
                # 1. A decision is needed. Call MCTS.
                assignments, root, state_vec = self.mcts_policy.decide(
                    net, net.t, free_servers
                )
                
                # 2. Get the training data
                policy_target = self.mcts_policy.get_policy_target(
                    root, self.config["temperature"]
                )
                
                # 3. Store for later
                episode_history.append((state_vec, policy_target))
                
                if not assignments:
                    # MCTS returned "NULL" action
                    break # Exit the 'while free_servers' loop
                
                # 4. Apply the *single* assignment
                for srv, q in assignments.items():
                    if len(q) == 0 or srv.busy: continue
                    job = q.pop()
                    dep_time = srv.start_service(job, net.t)
                    st_id = q.station_id
                    st = net.stations[st_id]
                    server_idx = st.servers.index(srv)
                    net.schedule(dep_time, EventType.DEPARTURE, {"station_id": st_id, "server_idx": server_idx})

                # 5. Check for more free servers *immediately*
                free_servers = net._get_free_servers()
                # Loop continues until all servers are assigned
        
        # 4. Episode is over, get final outcome (z)
        final_outcome_z, mean_sojourn = self.get_final_outcome(net)
        
        # 5. Add all steps to replay buffer with the final outcome
        if not episode_history:
            print("Warning: Episode ended with 0 decisions made.")
            return 0.0

        for state, policy_target in episode_history:
            self.replay_buffer.push(state, policy_target, final_outcome_z)
            
        return final_outcome_z, mean_sojourn

    def train_step(self) -> Optional[float]:
        """Samples a batch and performs one backprop step."""
        if len(self.replay_buffer) < self.config["batch_size"]:
            return None 

        states, policy_targets, value_targets = self.replay_buffer.sample(
            self.config["batch_size"]
        )
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        policy_targets = torch.tensor(policy_targets, dtype=torch.float32).to(self.device)
        value_targets = torch.tensor(value_targets, dtype=torch.float32).to(self.device)
        
        self.model.train() 
        policy_logits, values = self.model(states)
        
        value_loss = F.mse_loss(values, value_targets)
        policy_loss = F.cross_entropy(policy_logits, policy_targets)
        total_loss = value_loss + policy_loss # weight should not be equal
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

    def run_training_loop_parallel(self, num_workers=4):
        """
        Runs the training loop using Multiprocessing.
        """
        # Import inside function to avoid circular dependency
        from mcts_worker import run_episode_worker 

        print(f"\n--- PARALLEL TRAINING: {self.config['num_train_loops']} loops x {self.config['episodes_per_loop']} episodes ---")
        print(f"Workers: {num_workers}")
        
        ctx = mp.get_context('spawn')
        
        for i in range(self.config["num_train_loops"]):
            print(f"\n--- LOOP {i+1}/{self.config['num_train_loops']} ---")
            
            # 1. Prepare Data
            cpu_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            
            worker_args = []
            for j in range(self.config["episodes_per_loop"]):
                seed = random.randint(0, 10000000)
                snapshot = random.choice(self.start_states) if self.start_states else None
                # Add worker ID 'j' for logging
                worker_args.append((j, cpu_model_state, self.config, seed, snapshot))
            
            # 2. Run Parallel Generation (UPDATED FOR LIVE PROGRESS)
            episode_results = []
            
            with ctx.Pool(processes=num_workers) as pool:
                # Use imap_unordered to process results AS THEY ARRIVE
                total_eps = self.config["episodes_per_loop"]
                completed_count = 0
                
                # Create iterator
                results_iterator = pool.imap_unordered(run_episode_worker, worker_args)
                
                for history, score, raw_metric in results_iterator:
                    episode_results.append((history, score, raw_metric))
                    completed_count += 1
                    
                    # Print progress bar on the SAME LINE (\r)
                    print(f"  > Episode {completed_count}/{total_eps} finished. (Last Score: {score:.3f}, Size: {raw_metric:.2f})", end='\r')
            
            # Print new line after loop finishes so we don't overwrite the progress bar
            print("") 
            
            # 3. Process Results (Same as before)
            avg_score = 0
            avg_raw = 0
            total_new_samples = 0
            
            for history, score, raw_metric in episode_results:
                avg_score += score
                avg_raw += raw_metric
                
                for state_vec, policy_target in history:
                    self.replay_buffer.push(state_vec, policy_target, score)
                    total_new_samples += 1
            
            print(f"  Generated {total_new_samples} samples.")
            print(f"  Avg Score: {avg_score / len(episode_results):.4f}")
            print(f"  Avg Sys Size: {avg_raw / len(episode_results):.4f}")
            
            # 4. Train
            avg_loss = 0
            train_steps = 0
            for _ in range(self.config["train_steps_per_loop"]):
                loss = self.train_step()
                if loss:
                    avg_loss += loss
                    train_steps += 1
            
            if train_steps > 0:
                print(f"  Avg Loss: {avg_loss / train_steps:.4f}")
    
       # --- Helper to calculate System Size from a Network object ---
    def calculate_mean_system_size(self, net: Network) -> float:
        """
        Calculates the time-averaged number of jobs in the system.
        Formula: (Integral of Q(t) + S(t)) / Total Time
        """
        total_area = 0.0
        
        # Sum area of all queues and servers across all stations
        for st in net.stations.values():
            # Add Queue Area (waiting jobs * time)
            total_area += sum(st._ql_area.values())
            # Add Service Area (busy servers * time)
            total_area += st._sl_area
            
        elapsed_time = max(net.t, 1e-12) # Avoid division by zero
        return total_area / elapsed_time

    def save_model(self, filepath="mcts_alphazero_model.pth"):
        """Saves the trained model weights."""
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    # --- Updated Evaluation Method ---
    def evaluate_final_policy(self, num_episodes=10) -> Dict[str, float]:
        """
        Runs evaluative episodes and reports MEAN SYSTEM SIZE (L).
        """
        print(f"\n--- Starting Final Evaluation (Metric: System Size, {num_episodes} episodes) ---")
        
        # 1. Disable Exploration for now
        old_temp = self.config["temperature"]
        old_eps = self.config["dirichlet_epsilon"]
        self.config["temperature"] = 0.0 
        self.config["dirichlet_epsilon"] = 0.0
        
        self.mcts_policy.model.eval()
        
        system_sizes = []
        total_completed = 0
        
        for i in range(num_episodes):
            net = ExtendedSixClassNetwork(
                policy=self.mcts_policy,
                L=self.config["L"],
                seed=random.randint(100000, 999999)
            )
            
            if not net._seeded:
                for ap in net.arrivals:
                    t_next = ap.schedule_next(net.t)
                    net.schedule(t_next, EventType.ARRIVAL, {"ap": ap})
                net._seeded = True
            
            
            while net._event_q:
                if net._event_q[0].time > self.config["sim_run_duration"]:
                    break

                print(net.t)
                ev = heapq.heappop(net._event_q)
                
                dt = ev.time - net.t
                if dt > 0:
                    for st in net.stations.values():
                        for qid, q in st.queues.items():
                            st._ql_area[qid] += len(q) * dt
                        num_busy = sum(1 for srv in st.servers if srv.busy)
                        st._sl_area += num_busy * dt
                net.t = ev.time
    
                if ev.type == EventType.ARRIVAL:
                    net._on_arrival(ev.payload["ap"])
                elif ev.type == EventType.DEPARTURE:
                    net._on_departure(ev.payload["station_id"], ev.payload["server_idx"])
    
                free_servers = net._get_free_servers()
                while free_servers:
                    assignments, _, _ = self.mcts_policy.decide(net, net.t, free_servers)
                    if not assignments: break
                    for srv, q in assignments.items():
                        if len(q) == 0 or srv.busy: continue
                        job = q.pop()
                        srv.start_service(job, net.t)
                        st_id = q.station_id
                        server_idx = net.stations[st_id].servers.index(srv)
                        net.schedule(net.t + srv.service_sampler(job), EventType.DEPARTURE, {"station_id": st_id, "server_idx": server_idx})
                    free_servers = net._get_free_servers()

                if net.t % 2000 == 0:
                    print(net.t)
                    print(self.calculate_mean_system_size(net))
            
            # --- Calculate System Size ---
            mean_sys_size = self.calculate_mean_system_size(net)
            system_sizes.append(mean_sys_size)
            
            total_completed += net.completed_jobs
            print(f"  Eval Episode {i+1}: Mean System Size = {mean_sys_size:.4f}, Jobs = {net.completed_jobs}")
    
        # 2. Restore Config
        self.config["temperature"] = old_temp
        self.config["dirichlet_epsilon"] = old_eps
        
        avg_sys_size = np.mean(system_sizes)
        
        results = {
            "mean_system_size": avg_sys_size,
            "std_system_size": np.std(system_sizes),
            "total_jobs": total_completed
        }
        
        print("\n--- Evaluation Results ---")
        print(f"Average Mean System Size: {avg_sys_size:.4f}")
        return results


# In[3]:


def generate_lbfs_start_states(num_states=200, warmup=5000.0, separation=500.0):
    """
    Runs an LBFS simulation and captures snapshots of the system 
    at steady state to use as training starting points.
    """
    print(f"--- Generating {num_states} Warm Start States using LBFS ---")
    
    # 1. Setup LBFS Environment
    policy = LBFSPolicy()
    net = ExtendedSixClassNetwork(policy=policy, L=2, seed=42)
    
    # 2. Warmup Phase (Get to steady state)
    print(f"Warming up for {warmup} time units...")
    net.run(until_time=warmup)
    
    start_states = []
    
    # 3. Sampling Phase
    for i in range(num_states):
        # Run for a bit to change the state (decorrelate samples)
        run_until = net.t + separation
        net.run(until_time=run_until)
        
        # Clone the state
        # Note: We must clone it so we have a frozen copy
        snapshot = net.clone()
        start_states.append(snapshot)
        
        if (i+1) % 50 == 0:
            print(f"  Captured {i+1}/{num_states} states...")
            
    print("Generation Complete.")
    return start_states

# Generate them (this takes a few seconds/minutes)
# 200 states is usually enough; we will pick randomly from them


# In[ ]:




