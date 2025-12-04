#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import heapq
import random
from mcts_NetworkClasses import ExtendedSixClassNetwork, EventType
from mcts_model import MCTS_Policy, AlphaZeroNN

def run_episode_worker(args):
    """
    Runs ONE training episode in a separate process.
    Fixed to handle Relative Time correctly for Warm Starts.
    """
    worker_id, model_state, config, seed, start_snapshot = args
    
    # 1. Reconstruct Local Environment
    device = torch.device("cpu")
    
    # ... (Model/Policy setup remains the same) ...
    temp_policy = MCTS_Policy(None, config)
    action_space = len(temp_policy.master_action_list)
    state_size = config["MAX_QUEUES_STATE"]
    
    local_model = AlphaZeroNN(state_size, action_space).to(device)
    local_model.load_state_dict(model_state)
    local_model.eval()
    
    local_config = config.copy()
    local_config["device"] = device
    local_policy = MCTS_Policy(local_model, local_config)
    
    # 2. Setup Network
    if start_snapshot:
        net = start_snapshot.clone()
        net.policy = local_policy
        
        # --- FIX STARTS HERE ---
        # Do NOT reset net.t to 0.0. The events are scheduled in the future (e.g. t=10000)
        # We must simply reset the *accumulators* so we only measure THIS episode.
        for st in net.stations.values():
            st._ql_area = {qid: 0.0 for qid in st.queues}
            st._sl_area = 0.0
        net.completed_jobs = 0
        net.sum_sojourn = 0.0
        
        # Capture the starting time
        start_time = net.t 
        
        net.rng = np.random.default_rng(seed)
    else:
        net = ExtendedSixClassNetwork(
            policy=local_policy,
            L=local_config["L"],
            seed=seed
        )
        start_time = 0.0

    episode_history = []
    
    if not net._seeded:
        for ap in net.arrivals:
            t_next = ap.schedule_next(net.t)
            net.schedule(t_next, EventType.ARRIVAL, {"ap": ap})
        net._seeded = True

    # 3. Run Event Loop using RELATIVE TIME
    # We run until 'start_time + duration'
    end_time = start_time + local_config["sim_run_duration"]
    decisions_in_episode = 0
    while net._event_q:
        # Check against the relative end time
        if net.t > end_time:
            break

        ev = heapq.heappop(net._event_q)
        dt = ev.time - net.t
        
        # Stats update (Standard)
        if dt > 0:
            for st in net.stations.values():
                for qid, q in st.queues.items():
                    st._ql_area[qid] += len(q) * dt
                st._sl_area += sum(1 for s in st.servers if s.busy) * dt
        net.t = ev.time

        if ev.type == EventType.ARRIVAL:
            net._on_arrival(ev.payload["ap"])
        elif ev.type == EventType.DEPARTURE:
            net._on_departure(ev.payload["station_id"], ev.payload["server_idx"])

        # MCTS Decision
        free_servers = net._get_free_servers()
        while free_servers:
            assignments, root, state_vec = local_policy.decide(net, net.t, free_servers)
            if decisions_in_episode % 10 == 0: 
                policy_target = local_policy.get_policy_target(root, local_config["temperature"])
                episode_history.append((state_vec, policy_target))
                
            decisions_in_episode += 1
            
            if not assignments: break
            
            for srv, q in assignments.items():
                if len(q) == 0 or srv.busy: continue
                job = q.pop()
                srv.start_service(job, net.t)
                st_id = q.station_id
                server_idx = net.stations[st_id].servers.index(srv)
                net.schedule(net.t + srv.service_sampler(job), EventType.DEPARTURE, {"station_id": st_id, "server_idx": server_idx})
            free_servers = net._get_free_servers()

    # 4. Calculate Results
    # Calculate duration relative to this specific episode
    elapsed = max(net.t - start_time, 1e-12)
    total_area = sum(sum(st._ql_area.values()) + st._sl_area for st in net.stations.values())
    mean_sys_size = total_area / elapsed
    
    # Log-Space Normalization (Matches mcts_model.py)
    cat_val = local_config["CATASTROPHE_SOJOURN_TIME"]
    
    # Log Transform: log(1 + L)
    log_val = np.log1p(mean_sys_size)
    ref_log = np.log1p(cat_val)
    
    # Soft Normalization (No hard clip)
    final_score = 1.0 - (2.0 * (log_val / ref_log))
    
    return episode_history, float(final_score), mean_sys_size


# In[ ]:




