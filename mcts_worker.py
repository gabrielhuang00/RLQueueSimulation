#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import time
import numpy as np
import heapq
import random
from mcts_NetworkClasses import ExtendedSixClassNetwork, EventType
from mcts_model import MCTS_Policy, AlphaZeroNN

def run_episode_worker(args):
    wall_start = time.process_time()
    worker_id, model_state, config, seed, start_snapshot = args
    device = torch.device("cpu")
    
    temp_policy = MCTS_Policy(None, config)
    action_space = len(temp_policy.master_action_list)
    state_size = config["MAX_QUEUES_STATE"] + config["L"]
    
    local_model = AlphaZeroNN(state_size, action_space).to(device)
    local_model.load_state_dict(model_state)
    local_model.eval()
    
    local_config = config.copy()
    local_config["device"] = device
    local_policy = MCTS_Policy(local_model, local_config)
    
    if start_snapshot:
        net = start_snapshot.clone()
        net.policy = local_policy
        for st in net.stations.values():
            st._ql_area = {qid: 0.0 for qid in st.queues}
            st._sl_area = 0.0
        net.completed_jobs = 0
        net.sum_sojourn = 0.0
        start_time = net.t 
        net.rng = np.random.default_rng(seed)
    else:
        net = ExtendedSixClassNetwork(policy=local_policy, L=local_config["L"], seed=seed)
        start_time = 0.0

    episode_history = []
    
    if not net._seeded:
        for ap in net.arrivals:
            t_next = ap.schedule_next(net.t)
            net.schedule(t_next, EventType.ARRIVAL, {"ap": ap})
        net._seeded = True

    end_time = start_time + local_config["sim_run_duration"]
    decisions_in_episode = 0

    while net._event_q:
        if net.t > end_time: break

        ev = heapq.heappop(net._event_q)
        dt = ev.time - net.t
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

        free_servers = net._get_free_servers()
        while free_servers:
            assignments, root, state_vec, root_q, action_mask = local_policy.decide(net, net.t, free_servers)
            
            #if decisions_in_episode % 5 == 0: 
            policy_target = local_policy.get_policy_target(root, local_config["temperature"])
            episode_history.append((state_vec, policy_target, root_q, action_mask))
            
            root.detach()

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

    elapsed = max(net.t - start_time, 1e-12)
    total_area = sum(sum(st._ql_area.values()) + st._sl_area for st in net.stations.values())
    mean_sys_size = total_area / elapsed
    final_score = temp_policy._compute_score(mean_sys_size)
    wall_duration = time.process_time() - wall_start

    
    return episode_history, float(final_score), mean_sys_size, wall_duration


# In[ ]:




