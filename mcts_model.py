#!/usr/bin/env python
# coding: utf-8

# In[22]:


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
    "L": 2,                       
    "MAX_QUEUES_STATE": 6,       
    "MAX_STATIONS": 2,            
    "MAX_QUEUES_PER_STATION": 3,  
    
    "learning_rate": 0.001,
    "buffer_size": 250000,
    "batch_size": 256, 
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),

    "mcts_simulations": 200,   # This is low. But we have computational bottleneck.
    "c_puct": 1.5,
    "mcts_sim_horizon_s": 100.0,
    "temperature": 1.0,
    "dirichlet_alpha": 0.3,   
    "dirichlet_epsilon": 0.15, 
    "discount_factor": 0.97,  # Experiment.
    
    "num_train_loops": 100,
    "episodes_per_loop": 30,      
    "train_steps_per_loop": 150,  # <--- INCREASED per recommendation
    "sim_run_duration": 10000.0,    
    "CATASTROPHE_SOJOURN_TIME": 80.0, 
    "seed": 1
}

# --- 1. The Neural Network ---
class AlphaZeroNN(nn.Module):
    def __init__(self, state_size, action_space_size):
        super(AlphaZeroNN, self).__init__()
        self.state_size = state_size
        self.action_space_size = action_space_size
        self.body = nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_space_size)
        )
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, state):
        body_out = self.body(state)
        policy_logits = self.policy_head(body_out)
        value = self.value_head(body_out)
        return policy_logits, value

# --- 2. The Replay Buffer ---
class ReplayBuffer:
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
        self.children: Dict[int, MCTSNode] = {} 
        
        self.N = 0      
        self.W = 0.0    
        self.Q = 0.0    
        self.P = prior_p  
        
        self._state_vector: Optional[np.ndarray] = None
        self._is_terminal: bool = False

    def detach(self):
        """
        Manually breaks the cycle between parent and children
        to allow instant garbage collection.
        """
        for child in self.children.values():
            child.detach()
        self.children.clear()
        self.parent = None
        self._state_vector = None


    def get_state_vector(self, sim_net: Network, max_queues: int, context_server_idx: int = -1, L: int = 2) -> np.ndarray:
        if self._state_vector is None:
            # 1. Get Queue Lengths (as a standard list)
            q_lengths_dict = sim_net.total_queue_lengths()
            ordered_q_lengths = OrderedDict(sorted(q_lengths_dict.items()))
            state = [length for length in ordered_q_lengths.values()]
            
            # 2. Pad the List (List + List is safe!)
            # We pad with 0s here. Since log1p(0) = 0, this works perfectly.
            padding = [0] * (max_queues - len(state))
            padded_state = state + padding
            
            # 3. Log Normalize the Padded List
            # This creates [log(q1), log(q2)..., 0.0, 0.0]
            log_queues = np.log1p(np.array(padded_state, dtype=np.float32))

            # 4. Create Server Context (One-Hot)
            server_one_hot = np.zeros(L, dtype=np.float32)
            if context_server_idx >= 0 and context_server_idx < L:
                server_one_hot[context_server_idx] = 1.0
                
            # 5. Concatenate: [Queues..., Padding..., Servers...]
            self._state_vector = np.concatenate([log_queues, server_one_hot])
            
        return self._state_vector
            
        return self._state_vector
    def expand(self, policy_probs: np.ndarray):
        for action_idx, prob in enumerate(policy_probs):
            if prob > 0: 
                if action_idx not in self.children:
                    self.children[action_idx] = MCTSNode(parent=self, prior_p=prob)

    def select_child_puct(self, c_puct: float) -> Tuple[int, MCTSNode]:
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

    def update_stats(self, value: float):
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

# --- 4. The MCTS Policy ---
class MCTS_Policy(SchedulingPolicy):
    def __init__(self, model: AlphaZeroNN, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = config["device"]
        self.master_action_list = self._build_master_action_list()
        self.action_to_idx = {action: i for i, action in enumerate(self.master_action_list)}
        self.idx_to_action = {i: action for i, action in enumerate(self.master_action_list)}
        self.NULL_ACTION_IDX = self.action_to_idx[("NULL", "NULL")]

    def _build_master_action_list(self) -> List[Tuple[str, str]]:
        actions = []
        for i in range(1, self.config["L"] + 1):
            station_id = f"S{i}"
            server_id = f"{station_id}-s0"
            for k in range(1, 4):
                class_id = 3 * (i - 1) + k
                queue_id = f"Q{class_id}"
                actions.append((server_id, queue_id))
        actions.append(("NULL", "NULL"))
        return actions

    def _get_server_context_idx(self, station_id: str) -> int:
        # Assumes station_id is "S1", "S2", etc.
        # Returns 0 for S1, 1 for S2...
        return int(station_id.replace("S", "")) - 1

    
    def _get_action_mask(self, sim_net: Network, free_servers: Dict[str, List[Server]]) -> np.ndarray:
        mask = np.zeros(len(self.master_action_list), dtype=int)
        
        # 1. Identify which queues have jobs
        non_empty_queues = {
            (sid, qid) for sid, st in sim_net.stations.items()
            for qid, q in st.queues.items() if len(q) > 0
        }
        
        found_legal_action = False
        
        # 2. Find all valid server-to-queue assignments
        for station_id, srvs in free_servers.items():
            for srv in srvs:
                for action_idx, (srv_id, qid) in enumerate(self.master_action_list):
                    # STRICT MATCH: Ensure Action Server ID == Real Server ID
                    if srv.server_id == srv_id:
                        if (station_id, qid) in non_empty_queues:
                            mask[action_idx] = 1
                            found_legal_action = True
                            
        # 3. ONLY allow NULL if we found NO other legal actions
        # This forces the policy to be "Work Conserving"
        if not found_legal_action:
            mask[self.NULL_ACTION_IDX] = 1
            
        return mask
    
    def _compute_score(self, mean_sys_size: float) -> float:
        cat_val = self.config["CATASTROPHE_SOJOURN_TIME"]
        log_val = np.log1p(mean_sys_size)
        ref_log = np.log1p(cat_val)
        final_score = 1.0 - (2.0 * (log_val / ref_log))
        return final_score

    def decide(self, net: Network, t: float, free_servers: Dict[str, List[Server]]) -> Tuple[Dict[Server, Queue], MCTSNode, np.ndarray, float]:
        #Find server to assign.
        server_to_assign = None
        station_id_for_server = None 
        for st_id, srvs in free_servers.items():
            if srvs:
                server_to_assign = srvs[0]
                station_id_for_server = st_id     
                break

        #Check if there are any jobs.
        target_station = net.stations[station_id_for_server]
        target_station_id = station_id_for_server
        jobs_at_station = sum(len(q) for q in target_station.queues.values())
        
        if jobs_at_station == 0:
            # No jobs for this specific server -> Must Wait (NULL).
            
            # Create a dummy root for return types
            root = MCTSNode()
            # Get the correct context index for the neural net input (e.g., S1 -> 0, S2 -> 1)
            root_srv_idx = self._get_server_context_idx(target_station_id)
            state_vec = root.get_state_vector(net, self.config["MAX_QUEUES_STATE"], root_srv_idx)
            
            # Return empty assignments immediately
            return {}, root, state_vec, 0.0

        #Check if there's actually a free server.
        if server_to_assign is None:
            root = MCTSNode()
            state_vec = root.get_state_vector(net, self.config["MAX_QUEUES_STATE"], root_srv_idx)
            return {}, root, state_vec, 0.0

        #Make sure the network is cloneable. 
        try:
            real_snapshot = net.clone()
        except:
            root = MCTSNode()
            state_vec = root.get_state_vector(net, self.config["MAX_QUEUES_STATE"], root_srv_idx)
            return {}, root, state_vec, 0.0

        #Get one-hot encoding for free station.
        root_srv_idx = -1
        if station_id_for_server:
            root_srv_idx = self._get_server_context_idx(station_id_for_server)

        root = MCTSNode()
        state_vec = root.get_state_vector(net, self.config["MAX_QUEUES_STATE"], root_srv_idx)
        search_free_servers = {station_id_for_server: [server_to_assign]}
        
        #rng_state = random.getstate()
        #np_rng_state = np.random.get_state()
        
        for _ in range(self.config["mcts_simulations"]):
            #random.setstate(rng_state)
            #np.random.set_state(np_rng_state)
            node = root
            sim_net = real_snapshot.clone()
            sim_t = t
            sim_free_servers = search_free_servers
            
            path = [node]
            step_rewards = []
            
            while node.children: 
                #print(node.children)
                action_idx, node = node.select_child_puct(self.config["c_puct"])
                path.append(node)
                #print("1. idx_to_action",self.idx_to_action[action_idx], action_idx)
                (sim_net, sim_t, sim_free_servers, is_terminal, step_reward) = self._run_sim_step(sim_net, self.idx_to_action[action_idx])
                step_rewards.append(step_reward)
                if is_terminal:
                    node._is_terminal = True
                    break
            
            value = 0.0 
            #print("terminal", node._is_terminal)
            if not node._is_terminal:
                
                leaf_srv_idx = -1
                if sim_free_servers:
                    # Pick the first available station/server at the leaf state
                    #print("THERE ARE FREE SERVERS", sim_free_servers)
                    first_st_id = next(iter(sim_free_servers))
                    leaf_srv_idx = self._get_server_context_idx(first_st_id)
                    
                leaf_state_vec = node.get_state_vector(sim_net, self.config["MAX_QUEUES_STATE"], leaf_srv_idx)
                state_tensor = torch.tensor(leaf_state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    policy_logits, value_tensor = self.model(state_tensor)
                policy_logits = policy_logits.squeeze(0)
                value = value_tensor.item()
                
                action_mask = self._get_action_mask(sim_net, sim_free_servers)
                policy_logits[action_mask == 0] = -torch.inf
                policy_probs = F.softmax(policy_logits, dim=0).cpu().numpy()
                #print("policy_probs", policy_probs)
                
                if node is root:
                    num_legal = np.count_nonzero(action_mask)
                    if num_legal > 0:
                        noise = np.random.dirichlet([self.config["dirichlet_alpha"]] * num_legal)
                        eps = self.config["dirichlet_epsilon"]
                        idx = 0
                        for i in range(len(policy_probs)):
                            if action_mask[i]:
                                policy_probs[i] = (1 - eps) * policy_probs[i] + eps * noise[idx]
                                idx += 1
                                
                node.expand(policy_probs)
            
            # Backprop with One-Step Reward
            gamma = self.config["discount_factor"]
            curr_val = value 
            node.update_stats(curr_val)
            for i in range(len(path) - 2, -1, -1):
                node = path[i]
                reward = step_rewards[i]
                curr_val = (1 - gamma) * reward + gamma * curr_val
                node.update_stats(curr_val)

        if not root.children:
            best_action_idx = self.NULL_ACTION_IDX
        else:
            best_action_idx = max(root.children, key=lambda idx: root.children[idx].N)
        
        best_action = self.idx_to_action[best_action_idx]
        assignments = {}
        srv_id, qid = best_action
        if srv_id != "NULL":
            srv = server_to_assign
            if srv.server_id == srv_id:
                assignments[srv] = net.stations[station_id_for_server].queues[qid]

        # --- Return root.Q as the value target over episode reward ---
        # We return root.Q (which contains the One-Step Reward integration)
        # instead of relying on the noisy episode outcome later.
        return assignments, root, state_vec, root.Q

    def _run_sim_step(self, sim_net: Network, action: Tuple[str, str]) -> Tuple[Network, float, Dict, bool, float]:
        t_start = sim_net.t
        start_areas = {st.station_id: sum(st._ql_area.values()) + st._sl_area for st in sim_net.stations.values()}
        
        action_policy = FixedActionPolicy(action)
        sim_net.policy = action_policy
        free_now = sim_net._get_free_servers()
        if free_now:
            assignments = action_policy.decide(sim_net, sim_net.t, free_now)
            #print("3. assignments:",assignments)
            for srv, q in assignments.items():
                if len(q) == 0 or srv.busy: continue
                job = q.pop()
                dep_time = srv.start_service(job, sim_net.t)
                st_id = q.station_id
                st = sim_net.stations[st_id]
                idx = st.servers.index(srv)
                #print("4. server_assigned")
                sim_net.schedule(dep_time, EventType.DEPARTURE, {"station_id": st_id, "server_idx": idx})
        
        (new_t, new_free) = sim_net.run_until_next_decision()
        
        dt = max(new_t - t_start, 1e-12)
        total_sys_size_integral = 0.0
        for st in sim_net.stations.values():
            end_area = sum(st._ql_area.values()) + st._sl_area
            total_sys_size_integral += (end_area - start_areas[st.station_id])
            
        avg_size = total_sys_size_integral / dt
        step_reward = self._compute_score(avg_size)
        #print(total_sys_size_integral, avg_size, dt)
        is_terminal = (not sim_net._event_q) or (new_t > t_start + self.config["mcts_sim_horizon_s"])
        return sim_net, new_t, new_free, is_terminal, step_reward

    def get_policy_target(self, root: MCTSNode, temperature: float) -> np.ndarray:
        policy_target = np.zeros(len(self.master_action_list))
        if not root.children:
            policy_target[self.NULL_ACTION_IDX] = 1.0
            return policy_target
        visit_counts = np.array([child.N for child in root.children.values()])
        action_indices = np.array([idx for idx in root.children.keys()])
        
        if temperature == 0:
            best = action_indices[np.argmax(visit_counts)]
            policy_target[best] = 1.0
        else:
            visit_counts = visit_counts ** (1.0 / temperature)
            psum = np.sum(visit_counts)
            if psum > 0:
                probs = visit_counts / psum
                for idx, prob in zip(action_indices, probs):
                    policy_target[idx] = prob
            else:
                for idx in action_indices: policy_target[idx] = 1.0/len(action_indices)
        return policy_target

class FixedActionPolicy(SchedulingPolicy):
    def __init__(self, action): self.srv_id, self.qid = action
    def decide(self, net, t, free_servers):
        assignments = {}
        if self.srv_id == "NULL":
            return assignments
        for st_id, srvs in free_servers.items():
            for srv in srvs:
                if srv.server_id == self.srv_id:
                    assignments[srv] = net.stations[st_id].queues[self.qid]
                    #print("2. matched!")
                    return assignments
        #print("2. No match", srv.server_id, self.srv_id)
        return assignments

class Trainer:
    def __init__(self, config: Dict[str, Any], model = None, start_states=None):
        self.config = config
        self.device = config["device"]
        self.seed = config["seed"]
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        temp_policy = MCTS_Policy(None, self.config)
        self.action_space_size = len(temp_policy.master_action_list)
        self.state_size = self.config["MAX_QUEUES_STATE"] + self.config["L"]
        self.start_states = start_states
        
        if model is None:
            self.model = AlphaZeroNN(self.state_size, self.action_space_size).to(self.device)
        else: 
            self.model = model
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        self.replay_buffer = ReplayBuffer(config["buffer_size"])
        self.mcts_policy = MCTS_Policy(self.model, self.config)
        self.mcts_policy.model.eval()

    def calculate_mean_system_size(self, net: Network) -> float:
        total = sum(sum(st._ql_area.values()) + st._sl_area for st in net.stations.values())
        return total / max(net.t, 1e-12)

    def save_model(self, filepath="mcts_alphazero_model.pth"):
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def train_step(self) -> Optional[float]:
        if len(self.replay_buffer) < self.config["batch_size"]: return None 
        states, policy_targets, value_targets = self.replay_buffer.sample(self.config["batch_size"])
        
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        policy_targets = torch.tensor(policy_targets, dtype=torch.float32).to(self.device)
        value_targets = torch.tensor(value_targets, dtype=torch.float32).to(self.device)
        
        self.model.train() 
        policy_logits, values = self.model(states)
        
        value_loss = F.mse_loss(values, value_targets)
        policy_loss = F.cross_entropy(policy_logits, policy_targets)
        total_loss = value_loss + policy_loss 
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def run_training_loop_parallel(self, num_workers=4):
        from mcts_worker import run_episode_worker 
        print(f"\n--- PARALLEL TRAINING: {self.config['num_train_loops']} loops x {self.config['episodes_per_loop']} eps ---")
        ctx = mp.get_context('spawn')
        
        for i in range(self.config["num_train_loops"]):
            print(f"\n--- LOOP {i+1}/{self.config['num_train_loops']} ---")
            cpu_model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            worker_args = []
            for j in range(self.config["episodes_per_loop"]):
                seed = random.randint(0, 10000000)
                snapshot = random.choice(self.start_states) if self.start_states else None
                worker_args.append((j, cpu_model_state, self.config, seed, snapshot))
            
            episode_results = []
            with ctx.Pool(processes=num_workers) as pool:
                total_eps = self.config["episodes_per_loop"]
                completed_count = 0
                results_iterator = pool.imap_unordered(run_episode_worker, worker_args)
                
                for history, score, raw_metric, duration in results_iterator:
                    episode_results.append((history, score, raw_metric, duration))
                    completed_count += 1
                    print(f"  > Episode {completed_count}/{total_eps} fin. (Score: {score:.3f}, Size: {raw_metric:.2f})", end='\r')
            print("") 
            
            avg_score = 0
            avg_raw = 0
            total_samples = 0
            
            for history, score, raw_metric, duration in episode_results:
                avg_score += score
                avg_raw += raw_metric
                for state_vec, policy_target, root_q in history:
                    self.replay_buffer.push(state_vec, policy_target, root_q) #conflicted root_q or score.
                    total_samples += 1
                    
            total_duration = sum(d for _, _, _, d in episode_results)
            avg_duration = total_duration / len(episode_results)
            
            print(f"  Generated {total_samples} samples.")
            print(f"  Avg Score: {avg_score/len(episode_results):.4f}")
            print(f"  Avg Sys Size: {avg_raw/len(episode_results):.4f}")
            print(f"  Avg Episode Time: {avg_duration:.2f}s") 
            avg_loss = 0
            t_steps = 0
            for _ in range(self.config["train_steps_per_loop"]):
                loss = self.train_step()
                if loss:
                    avg_loss += loss
                    t_steps += 1
            if t_steps > 0: print(f"  Avg Loss: {avg_loss/t_steps:.4f}")
            
            if (i+1) % 10 == 0: self.save_model(f"model_loop_{i+1}.pth")

    # (Sequential debug method omitted for brevity, ensure get_final_outcome has self)
    def get_final_outcome(self, net: Network) -> Tuple[float, float]:
        mean_sys_size = self.calculate_mean_system_size(net) 
        log_val = np.log1p(mean_sys_size)
        ref_log = np.log1p(self.config["CATASTROPHE_SOJOURN_TIME"])
        final_score = 1.0 - (2.0 * (log_val / ref_log))
        return float(final_score), mean_sys_size


# In[14]:


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


# In[24]:


temp_policy = MCTS_Policy(None, CONFIG)

