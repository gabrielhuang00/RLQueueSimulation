#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from collections import deque
import heapq
import math
import numpy as np
from collections import defaultdict
import copy 
import matplotlib.pyplot as plt # 

# -------- Events --------

class EventType(Enum):
    ARRIVAL = auto()
    DEPARTURE = auto()

@dataclass(order=True)
class Event:
    time: float
    order: int
    type: EventType = field(compare=False)
    payload: Dict[str, Any] = field(compare=False, default_factory=dict)
    
    def clone(self) -> Event:
        # Dataclasses are easy to clone
        return copy.deepcopy(self)


# -------- Core domain --------

@dataclass
class Job:
    id: int
    cls: str
    t_arrival: float
    q_arrival: float
    t_service_start: Optional[float] = None
    t_departure: Optional[float] = None
    # optional: path trace [(station_id, queue_id, t_enter, t_start, t_leave)]
    trace: List[Tuple[str, str, float, Optional[float], Optional[float]]] = field(default_factory=list)

    def clone(self) -> Job:
        # Dataclasses are easy to clone
        return copy.deepcopy(self)


class Queue:
    """
    FIFO queue by default. Replace with a priority queue if you need.
    """
    def __init__(self, station_id: str, queue_id: str):
        self.station_id = station_id
        self.queue_id = queue_id
        self._buf: Deque[Job] = deque()

    def push(self, job: Job, t: float) -> None:
        job.trace.append((self.station_id, self.queue_id, t, None, None))
        self._buf.append(job)

    def pop(self) -> Optional[Job]:
        if not self._buf:
            return None
        return self._buf.popleft()

    def __len__(self) -> int:
        return len(self._buf)

    def peek(self) -> Optional[Job]:
        return self._buf[0] if self._buf else None

    def peek_n(self, n: int) -> Optional[Job]:
        # n = 0 -> head, 1 -> second, ...
        if n < 0 or n >= len(self._buf):
            return None
        # deque supports O(n) indexing; n is tiny (<= #free servers), so this is fine
        return self._buf[n]

    def clone(self) -> Queue:
        new_q = Queue(self.station_id, self.queue_id)
        # We must clone each individual job in the buffer
        new_q._buf = deque([job.clone() for job in self._buf])
        return new_q


class Server:
    """
    Single server (non-preemptive). Station owns multiple of these.
    """
    def __init__(self, server_id: str, service_sampler: Callable[[Job], float]):
        self.server_id = server_id
        self.service_sampler = service_sampler
        self.busy: bool = False
        self.job: Optional[Job] = None
        self.t_busy_until: float = math.inf

    def start_service(self, job: Job, t: float) -> float:
        """Start service and return departure time."""
        assert not self.busy
        self.busy = True
        self.job = job
        job.t_service_start = t
        # mark trace start time
        if job.trace and job.trace[-1][3] is None:
            st, qid, t_enter, _, t_leave = job.trace[-1]
            job.trace[-1] = (st, qid, t_enter, t, t_leave)

        s = max(0.0, float(self.service_sampler(job)))
        dep_time = t + s
        self.t_busy_until = dep_time
        return dep_time

    def complete(self, t: float):
        assert self.busy and self.job is not None
        job = self.job
        # mark trace leave time for this stage
        if job.trace and job.trace[-1][4] is None:
            st, qid, t_enter, t_start, _ = job.trace[-1]
            job.trace[-1] = (st, qid, t_enter, t_start, t)
        self.busy = False
        self.job = None
        self.t_busy_until = math.inf
        return job

    def clone(self) -> Server:
        # Note: service_sampler is a function (closure) and is copied
        # by reference. This is what we want.
        new_s = Server(self.server_id, self.service_sampler)
        new_s.busy = self.busy
        # We must clone the job that is currently in service
        new_s.job = self.job.clone() if self.job else None
        new_s.t_busy_until = self.t_busy_until
        return new_s


class Station:
    """
    Station with c servers and multiple queues.
    """
    def __init__(
        self,
        station_id: str,
        queues: Dict[str, Queue],
        servers: List[Server],
    ):
        self.station_id = station_id
        self.queues = queues         # queue_id -> Queue
        self.servers = servers       # list of Server

    def free_servers(self) -> List[Server]:
        return [s for s in self.servers if not s.busy]

    def total_queue_len(self) -> int:
        return sum(len(q) for q in self.queues.values())

    def queue_lengths(self) -> Dict[str, int]:
        return {qid: len(q) for qid, q in self.queues.items()}

    def clone(self) -> Station:
        new_queues = {qid: q.clone() for qid, q in self.queues.items()}
        new_servers = [s.clone() for s in self.servers]
        new_st = Station(self.station_id, new_queues, new_servers)
        
        if hasattr(self, '_sl_area'):
            new_st._sl_area = self._sl_area
        if hasattr(self, '_ql_area'):
            new_st._ql_area = dict(self._ql_area) # shallow copy dict
        return new_st


# -------- Arrivals and Routing --------

class ArrivalProcess:
    """
    External arrival process targeting (station_id, queue_id).
    For exponential inter-arrivals: sampler() returns Exp(λ).
    """
    def __init__(
        self,
        target_station: str,
        target_queue: str,
        interarrival_sampler: Callable[[], float],
        job_class: str,
    ):
        self.station_id = target_station
        self.queue_id = target_queue
        self.interarrival_sampler = interarrival_sampler
        self.job_class = job_class
        self.next_time: float = 0.0

    def schedule_next(self, t_now: float) -> float:
        self.next_time = t_now + max(0.0, float(self.interarrival_sampler()))
        return self.next_time

    def clone(self) -> ArrivalProcess:
        # Copying the sampler by reference is fine, but we must
        # create a new object to copy the 'next_time' state.
        new_ap = ArrivalProcess(
            self.station_id,
            self.queue_id,
            self.interarrival_sampler,
            self.job_class
        )
        new_ap.next_time = self.next_time
        return new_ap


class SchedulingPolicy:
    def decide(
        self,
        net, # NetworkLike
        t: float,
        free_servers: Dict[str, List[Server]], # Note: my MCTS needs Dict, not List
    ) -> Dict[Server, Queue]:
        raise NotImplementedError


# -------- Network (orchestrator) --------

class Network:
    """
    Continuous-time event-driven simulator.
    - Manages event list
    - Calls policy at decision epochs
    - Tracks metrics (e.g., sojourn times)
    """
    def __init__(
        self,
        stations: Dict[str, Station],
        arrivals: List[ArrivalProcess],
        router: Any, # Router class
        policy: SchedulingPolicy,
        rng: Optional[np.random.Generator] = None,
    ):
        self.stations = stations                      # station_id -> Station
        self.arrivals = arrivals
        self.router = router
        self.policy = policy
        self.rng = rng or np.random.default_rng(0)

        self.t: float = 0.0
        self._event_q: List[Event] = []
        self._eid: int = 0
        self._job_id_seq: int = 0

        # metrics
        self.completed_jobs: int = 0
        self.sum_sojourn: float = 0.0
        self.exited_jobs: list = []
        self._seeded: bool = False
        
        for st in self.stations.values():
            st._sl_area = 0.0
            st._ql_area = {qid: 0.0 for qid in st.queues}

    # ---- time & events ----

    def schedule(self, time: float, etype: EventType, payload: Dict[str, Any]) -> None:
        self._eid += 1
        heapq.heappush(self._event_q, Event(time=time, order=self._eid, type=etype, payload=payload))

    def run(self, until_time: Optional[float] = None, until_jobs: Optional[int] = None) -> None:
        """
        Runs the simulation using the original event loop.
        """
        if not self._seeded:
            for ap in self.arrivals:
                t_next = ap.schedule_next(self.t)
                self.schedule(t_next, EventType.ARRIVAL, {"ap": ap})
            self._seeded = True
    
        while self._event_q:
            if until_time is not None and self._event_q[0].time > until_time:
                break
    
            ev = heapq.heappop(self._event_q)
    
            if until_jobs is not None and self.completed_jobs >= until_jobs:
                heapq.heappush(self._event_q, ev)
                break
    
            # Advance time and accumulate areas for both queues and servers
            dt = ev.time - self.t
            if dt > 0:
                for st in self.stations.values():
                    # Accumulate area for jobs in queues
                    for qid, q in st.queues.items():
                        st._ql_area[qid] += len(q) * dt
                    
                    # Accumulate area for jobs in service
                    num_busy_servers = sum(1 for srv in st.servers if srv.busy)
                    st._sl_area += num_busy_servers * dt

            self.t = ev.time
    
            # Handle event
            if ev.type == EventType.ARRIVAL:
                self._on_arrival(ev.payload["ap"])
            elif ev.type == EventType.DEPARTURE:
                self._on_departure(ev.payload["station_id"], ev.payload["server_idx"])
            else:
                raise RuntimeError("Unknown event type")
    
            # Scheduling decision after any state change
            self._decision_epoch()

    # ---- arrivals, departures ----

    def _on_arrival(self, ap: ArrivalProcess) -> None:
        job = Job(id=self._next_job_id(), cls=ap.job_class, t_arrival=self.t, q_arrival = self.t)
        st = self.stations[ap.station_id]
        q = st.queues[ap.queue_id]
        q.push(job, self.t)

        # schedule next external arrival
        t_next = ap.schedule_next(self.t)
        self.schedule(t_next, EventType.ARRIVAL, {"ap": ap})
    
    def _on_departure(self, station_id: str, server_idx: int) -> None:
        st = self.stations[station_id]
        srv = st.servers[server_idx]
        job = srv.complete(self.t)
        nxt = self.router.route(job, station_id, self.t)
        if nxt is None:
            job.t_departure = self.t
            self.completed_jobs += 1
            self.sum_sojourn += (job.t_departure - job.t_arrival)
            job.t_departure = self.t
            self.exited_jobs.append(job) 
        else:
            next_st_id, next_q_id = nxt
            job.q_arrival = self.t
            self.stations[next_st_id].queues[next_q_id].push(job, self.t)

    # ---- policy invocation ----

    def _decision_epoch(self) -> None:
        # [THIS METHOD IS UNCHANGED]
        free: Dict[str, List[Server]] = {}
        for st_id, st in self.stations.items():
            # Use the simple list comprehension from Station
            free_servers_at_st = st.free_servers()
            if free_servers_at_st:
                free[st_id] = free_servers_at_st

        if not free:
            return

        assignments = self.policy.decide(self, self.t, free_servers=free)

        # start service for each assigned server
        for srv, q in assignments.items():
            if len(q) == 0 or srv.busy:
                continue
            job = q.pop()
            dep_time = srv.start_service(job, self.t)
            st_id = q.station_id
            st = self.stations[st_id]
            server_idx = st.servers.index(srv)
            self.schedule(dep_time, EventType.DEPARTURE, {"station_id": st_id, "server_idx": server_idx})

    # ---- helpers ----
     
    def _next_job_id(self) -> int:
        self._job_id_seq += 1
        return self._job_id_seq

    # Convenience state accessors for policies
    def queue_length(self, station_id: str, queue_id: str) -> int:
        return len(self.stations[station_id].queues[queue_id])

    def station_queue_lengths(self, station_id: str) -> Dict[str, int]:
        st = self.stations[station_id]
        return {qid: len(q) for qid, q in st.queues.items()}

    def total_queue_lengths(self) -> Dict[Tuple[str, str], int]:
        out = {}
        for sid, st in self.stations.items():
            for qid, q in st.queues.items():
                out[(sid, qid)] = len(q)
        return out

    def mean_sojourn(self) -> float:
        return self.completed_jobs / self.sum_sojourn if self.completed_jobs else float("nan")

    # -----------------------------------------------------------------
    # --- ALL NEW METHODS FOR MCTS---
    # -----------------------------------------------------------------

    def clone(self, new_policy: Optional[SchedulingPolicy] = None) -> Network:
        """
        Creates a deep snapshot of the simulation state.
        The "config" (router, samplers, rng) is copied by reference.
        The "state" (time, events, queues, jobs) is cloned.
        """
        
        # 1. Clone all stations (which clones servers, queues, and jobs)
        new_stations = {sid: st.clone() for sid, st in self.stations.items()}
        
        # 2. Clone arrival processes (to copy 'next_time' state)
        new_arrivals = [ap.clone() for ap in self.arrivals]

        # 3. Create the new network
        # We re-use the router and rng by reference.
        # This is a "noisy" clone: all MCTS sims will share one RNG.
        # This is simpler than re-architecting the samplers.
        new_net = Network(
            stations=new_stations,
            arrivals=new_arrivals,
            router=self.router,
            policy=new_policy or self.policy, # Use new policy if given
            rng=np.random.default_rng(self.rng.integers(1_000_000_000))
        )
        
        # 4. Copy all state attributes
        new_net.t = self.t
        new_net._event_q = [ev.clone() for ev in self._event_q] # Must clone!
        new_net._eid = self._eid
        new_net._job_id_seq = self._job_id_seq
        new_net._seeded = self._seeded
        
        # 5. Copy all metric attributes
        new_net.completed_jobs = self.completed_jobs
        new_net.sum_sojourn = self.sum_sojourn
        new_net.exited_jobs = []
        
        # (Station-level metric areas were copied in station.clone())
        
        return new_net

    def _get_free_servers(self) -> Dict[str, List[Server]]:
            """Helper to find all free servers, grouped by station."""
            free: Dict[str, List[Server]] = {}
            for st_id, st in self.stations.items():
                free_servers_at_st = st.free_servers()
                if free_servers_at_st:
                    free[st_id] = free_servers_at_st
            return free # Returns {} if none are free

    def run_until_next_decision(self) -> Tuple[float, Dict[str, List[Server]]]:
            """
            The "stepper" function for MCTS.
            Runs the sim until the next decision epoch is reached.
            """
            
            # Seed if this is the very first step
            if not self._seeded:
                for ap in self.arrivals:
                    t_next = ap.schedule_next(self.t)
                    self.schedule(t_next, EventType.ARRIVAL, {"ap": ap})
                self._seeded = True
    
            while self._event_q:

                ev = heapq.heappop(self._event_q)
        
                # --- FIX: CALCULATE DT AND UPDATE AREAS ---
                dt = ev.time - self.t
                if dt > 0:
                    for st in self.stations.values():
                        # Accumulate area for jobs in queues
                        for qid, q in st.queues.items():
                            st._ql_area[qid] += len(q) * dt
                        
                        # Accumulate area for jobs in service
                        # (Optimized check: only count busy servers)
                        num_busy_servers = sum(1 for srv in st.servers if srv.busy)
                        st._sl_area += num_busy_servers * dt
                # ------------------------------------------
                # 3. Advance time
                self.t = ev.time
        
                # 4. Handle event
                if ev.type == EventType.ARRIVAL:
                    self._on_arrival(ev.payload["ap"])
                elif ev.type == EventType.DEPARTURE:
                    self._on_departure(ev.payload["station_id"], ev.payload["server_idx"])
                
                # 5. Check if a decision is now needed
                free_after_event = self._get_free_servers()
                if free_after_event:
                    return (self.t, free_after_event)
                
            return (self.t, {})


# -------- Utility samplers (Exp arrivals/services) --------
# [UNCHANGED - Included for completeness]

def exp_interarrival(rate: float, rng: np.random.Generator) -> Callable[[], float]:
    assert rate > 0
    return lambda: rng.exponential(1.0 / rate)

def exp_service(mu: float, rng: np.random.Generator) -> Callable[[Job], float]:
    assert mu > 0
    return lambda job: rng.exponential(1.0 / mu)


# In[5]:


# -------- Policy Implementations --------

class FIFOSysPolicy(SchedulingPolicy):
    def decide(self, net, t, free_servers):
        assignments: Dict[Server, Queue] = {}
        for st_id, srvs in free_servers.items():
            st = net.stations[st_id]
            used = defaultdict(int)
            for srv in srvs:
                best_q, best_time = None, float("inf")
                for q in st.queues.values():
                    j = q.peek_n(used[q])
                    if j and j.q_arrival < best_time:
                        best_q, best_time = q, j.q_arrival
                if best_q:
                    assignments[srv] = best_q
                    used[best_q] += 1
        return assignments


class FIFONetPolicy(SchedulingPolicy):
    def decide(self, net, t, free_servers):
        assignments: Dict[Server, Queue] = {}
        for st_id, srvs in free_servers.items():
            st = net.stations[st_id]
            used = defaultdict(int)
            for srv in srvs:
                best_q, best_time = None, float("inf")
                for q in st.queues.values():
                    j = q.peek_n(used[q])
                    if j and j.t_arrival < best_time:
                        best_q, best_time = q, j.t_arrival
                if best_q:
                    assignments[srv] = best_q
                    used[best_q] += 1
        return assignments


class MaxWeightByQLenPolicy(SchedulingPolicy):
    def decide(self, net, t, free_servers):
        assignments: Dict[Server, Queue] = {}
        for st_id, srvs in free_servers.items():
            station = net.stations[st_id]
            queue_lengths = {qid: len(q) for qid, q in station.queues.items() if len(q) > 0}
            if not queue_lengths:
                continue
            for srv in srvs:
                if not queue_lengths:
                    break
                qid = max(queue_lengths, key=queue_lengths.get)
                assignments[srv] = station.queues[qid]
                queue_lengths[qid] -= 1
                if queue_lengths[qid] <= 0:
                    del queue_lengths[qid]
        return assignments


class MaxWeightChePolicy(SchedulingPolicy):
    def decide(self, net, t, free_servers):
        assignments: Dict[Server, Queue] = {}
        for st_id, srvs in free_servers.items():
            st = net.stations[st_id]
            eff = { qid: (len(q), net.mu_for_queue_id(qid))
                    for qid, q in st.queues.items() if len(q) > 0 }
            for srv in srvs:
                if not eff:
                    break
                qid_star = max(eff, key=lambda qid: eff[qid][0] * eff[qid][1])
                assignments[srv] = st.queues[qid_star]
                x, mu = eff[qid_star]
                x -= 1
                if x <= 0:
                    del eff[qid_star]
                else:
                    eff[qid_star] = (x, mu)
        return assignments


class LBFSPolicy(SchedulingPolicy):
    def decide(self, net: Network, t: float, free_servers: Dict[str, List[Server]]) -> Dict[Server, Queue]:
        assignments: Dict[Server, Queue] = {}
        for st_id, srvs in free_servers.items():
            station = net.stations[st_id]
            best_q: Optional[Queue] = None
            max_cls_id = -1
            for q in station.queues.values():
                if len(q) > 0:
                    try:
                        cls_id = int(q.queue_id.replace("Q", ""))
                        if cls_id > max_cls_id:
                            max_cls_id = cls_id
                            best_q = q
                    except ValueError:
                        continue
            if best_q:
                jobs_to_assign = len(best_q)
                for srv in srvs:
                    if jobs_to_assign > 0:
                        assignments[srv] = best_q
                        jobs_to_assign -= 1
                    else:
                        break
        return assignments


# In[6]:


class ExtendedServiceSampler:
    """
    A lightweight callable object that handles service times.
    Replaces 'self.service_sampler' to avoid memory leaks during pickling.
    """
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.mu_rates = { 
            1: 1.0 / 8.0, 2: 1.0 / 2.0, 3: 1.0 / 4.0,
            4: 1.0 / 6.0, 5: 1.0 / 7.0, 6: 1.0 / 1.0,
        }

    def __call__(self, job: Job) -> float:
        class_idx = int(job.cls)
        # Map class to rate key (1..6)
        key = (class_idx - 1) % 6 + 1
        rate = self.mu_rates[key]
        return self.rng.exponential(1.0 / rate)

class ExtendedArrivalSampler:
    """
    A lightweight callable object that handles arrival times.
    """
    def __init__(self, rng: np.random.Generator, lam: float):
        self.rng = rng
        self.lam = lam

    def __call__(self) -> float:
        return self.rng.exponential(1.0 / self.lam)

class ExtendedSixClassRouter:
    """
    A standalone router class.
    """
    def __init__(self, L: int):
        self.L = L

    def route(self, job: Job, station_id: str, t: float) -> Optional[Tuple[str, str]]:
        station_num = int(station_id.replace("S", ""))
        class_num = int(job.cls)
        
        if station_num < self.L:
            next_class = class_num + 3
            next_station = station_num + 1
            job.cls = str(next_class)
            return (f"S{next_station}", f"Q{next_class}")
        elif station_num == self.L:
            if class_num == 3 * (self.L - 1) + 1:
                job.cls = "2"
                return ("S1", "Q2")
            else:
                return None
        return None

# --- 2. The Fixed Network Class ---

class ExtendedSixClassNetwork(Network):
    """
    "Extended Six-Class Queueing Network" from Figure 4
    of Dai and Gluzman (2022).
    """
    def __init__(self,
                 policy: SchedulingPolicy,
                 *,
                 L: int, # Number of stations
                 seed: int = 0,
                 ):
        if not L >= 2:
            raise ValueError("L must be an integer >= 2 for this network.")
        
        # Create a dedicated RNG
        rng = np.random.default_rng(seed)
        lam = 9.0 / 140.0
        
        # --- Create Standalone Samplers ---
        # We pass these to Servers/Arrivals instead of 'self.method'
        # This breaks the reference cycle to the Network object.
        svc_sampler = ExtendedServiceSampler(rng)
        arr_sampler = ExtendedArrivalSampler(rng, lam)

        stations: Dict[str, Station] = {}
        for i in range(1, L + 1):
            sid = f"S{i}"
            station_queues: Dict[str, Queue] = {}
            for k in range(1, 4):
                class_id = 3 * (i - 1) + k
                qid = f"Q{class_id}"
                station_queues[qid] = Queue(sid, qid)
            
            # Pass the functor object, NOT a bound method
            station_servers = [Server(f"{sid}-s0", svc_sampler)]
            stations[sid] = Station(sid, station_queues, station_servers)

        # Pass the functor object
        ap1 = ArrivalProcess("S1", "Q1", arr_sampler, job_class="1")
        ap3 = ArrivalProcess("S1", "Q3", arr_sampler, job_class="3")

        # Pass the router object
        router = ExtendedSixClassRouter(L)

        super().__init__(
            stations=stations,
            arrivals=[ap1, ap3],
            router=router,
            policy=policy,
            rng=rng,
        )
        self._params = dict(L=L, lam=lam, mu_rates=svc_sampler.mu_rates)

    # --- Metrics Helper (Unchanged) ---
    def run_and_get_batch_means_stats(
        self,
        warmup_time: float,
        num_batches: int,
        batch_duration: float,
        include_service: bool = True, 
    ) -> Dict[str, Any]:
        print(f"Running warmup for {warmup_time:.0f} time units...")
        self.run(until_time=self.t + warmup_time)
        print("Warmup complete. Starting batch means measurement...")
    
        batch_means_queue = []
        batch_means_system = []
    
        for _ in range(num_batches):
            # reset areas
            for st in self.stations.values():
                st._ql_area = {qid: 0.0 for qid in st.queues}
                st._sl_area = 0.0
    
            t0 = self.t
            self.run(until_time=t0 + batch_duration)
            elapsed = max(1e-12, self.t - t0)
    
            q_area = 0.0
            s_area = 0.0
            for st in self.stations.values():
                q_area += sum(st._ql_area.values())
                s_area += st._sl_area
    
            batch_means_queue.append(q_area / elapsed)
            batch_means_system.append((q_area + s_area) / elapsed)
    
        def agg(arr):
            m = float(np.mean(arr))
            sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            ci = 1.96 * sd / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
            return m, sd, ci

        mean_q, sd_q, ci_q = agg(batch_means_queue)
        mean_sys, sd_sys, ci_sys = agg(batch_means_system)
    
        reported_mean = mean_sys if include_service else mean_q
        reported_ci   = ci_sys   if include_service else ci_q
    
        print("Measurement complete.")
        return {
            "mean_jobs_in_system": mean_sys,
            "ci_half_width": ci_sys,
            "reported_mean": reported_mean,
            "reported_ci_half_width": reported_ci,
            "num_batches": num_batches,
        }


# In[ ]:




