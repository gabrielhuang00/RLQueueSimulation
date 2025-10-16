#!/usr/bin/env python
# coding: utf-8

# In[63]:


from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple
from collections import deque
import heapq
import math
import numpy as np
from collections import defaultdict

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


class Server:
    """
    Single server (non-preemptive). Station owns multiple of these.
    """
    #service sampler is a function that maps a job to a float.
    #def svc1(job):  # e.g., class-specific service rates
    #return rng.exponential(1/ (1.5 if job.cls=="A" else 1.0))
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


# -------- Policy interface --------

class SchedulingPolicy:
    """
    Centralized policy: at a decision epoch (e.g., after arrivals or a departure),
    assign jobs to some/all free servers across the network.

    Must return a mapping: (server) -> (queue) from which to take the next job.
    The simulator will pop a job from that queue and start it on that server.
    """
    def decide(
        self,
        net: NetworkLike,                  # duck-typed: see Network below
        t: float,
        free_servers: List[Tuple[Station, Server]],
    ) -> Dict[Server, Queue]:
        raise NotImplementedError


class FIFOSysPolicy(SchedulingPolicy):
    """
    Greedy: at each decision epoch, for each free server, take from the
    non-empty queue of its own station in a fixed order (or max length).
    """
    
    def decide(self, net, t, free_servers):
        assignments: Dict[Server, Queue] = {}
        for st_id, srvs in free_servers.items():
            st = net.stations[st_id]
            used = defaultdict(int)  # virtual pops per queue in this epoch
            for srv in srvs:
                best_q, best_time = None, float("inf")
                for q in st.queues.values():
                    j = q.peek_n(used[q])       # 0=head, 1=second, ...
                    if j and j.q_arrival < best_time:
                        best_q, best_time = q, j.q_arrival
                if best_q:
                    assignments[srv] = best_q
                    used[best_q] += 1
        return assignments


class FIFONetPolicy(SchedulingPolicy):
    """
    First-Come, First-Serve (FCFS) Policy - System-Wide.
    At each decision epoch, for each free server, this policy assigns the job
    that has been in the entire system the longest. This is determined by the
    job's original network entry time (t_arrival).
    """
    
    def decide(self, net, t, free_servers):
        assignments: Dict[Server, Queue] = {}
        for st_id, srvs in free_servers.items():
            st = net.stations[st_id]
            used = defaultdict(int)  # virtual pops per queue in this epoch
            for srv in srvs:
                best_q, best_time = None, float("inf")
                for q in st.queues.values():
                    j = q.peek_n(used[q])
                    # THE ONLY CHANGE IS HERE: use j.t_arrival instead of j.q_arrival
                    if j and j.t_arrival < best_time:
                        best_q, best_time = q, j.t_arrival
                if best_q:
                    assignments[srv] = best_q
                    used[best_q] += 1
        return assignments


class MaxWeightByQLenPolicy(SchedulingPolicy):
    """
    Station-local MaxWeight: for each free server at a station,
    assign it to the nonempty queue with the largest length.
    """
    def decide(self, net, t, free_servers):
        assignments: Dict[Server, Queue] = {}
        for st_id, srvs in free_servers.items():
            station = net.stations[st_id]

            # initialize queue lengths
            queue_lengths = {qid: len(q) for qid, q in station.queues.items() if len(q) > 0}
            if not queue_lengths:
                continue

            for srv in srvs:
                if not queue_lengths:  # all empty now
                    break

                # pick the queue_id with the largest length
                qid = max(queue_lengths, key=queue_lengths.get)
                assignments[srv] = station.queues[qid]

                # virtually consume one job from that queue
                queue_lengths[qid] -= 1
                if queue_lengths[qid] <= 0:
                    del queue_lengths[qid]

        return assignments

class LBFSPolicy(SchedulingPolicy):
    """
    Last-Buffer First-Serve (LBFS) Policy.
    At each station, this policy gives static priority to the non-empty queue
    with the highest class index.
    """
    def decide(self, net: Network, t: float, free_servers: Dict[str, List[Server]]) -> Dict[Server, Queue]:
        assignments: Dict[Server, Queue] = {}
        for st_id, srvs in free_servers.items():
            station = net.stations[st_id]

            # Find the best non-empty queue based on the highest class index
            best_q: Optional[Queue] = None
            max_cls_id = -1

            for q in station.queues.values():
                if len(q) > 0:
                    # Parse class ID from queue ID (e.g., "Q7" -> 7)
                    try:
                        cls_id = int(q.queue_id.replace("Q", ""))
                        if cls_id > max_cls_id:
                            max_cls_id = cls_id
                            best_q = q
                    except ValueError:
                        # Handle non-numeric queue IDs if they exist
                        continue
            
            # If a priority queue was found, assign all free servers to it
            if best_q:
                # Keep track of jobs assigned from the best queue in this epoch
                jobs_to_assign = len(best_q)
                for srv in srvs:
                    if jobs_to_assign > 0:
                        assignments[srv] = best_q
                        jobs_to_assign -= 1
                    else:
                        break # No more jobs in the priority queue
                        
        return assignments

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
        router: Router,
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

    # ---- time & events ----

    def schedule(self, time: float, etype: EventType, payload: Dict[str, Any]) -> None:
        self._eid += 1
        heapq.heappush(self._event_q, Event(time=time, order=self._eid, type=etype, payload=payload))

    def run(self, until_time: Optional[float] = None, until_jobs: Optional[int] = None) -> None:
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
                    if not hasattr(st, "_ql_area"):
                        st._ql_area = {qid: 0.0 for qid in st.queues}
                    for qid, q in st.queues.items():
                        st._ql_area[qid] += len(q) * dt
                    
                    # Accumulate area for jobs in service
                    if not hasattr(st, "_sl_area"):
                        st._sl_area = 0.0
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
        # realize a job and enqueue
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
            # final network exit -- record sojourn once
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
        # collect all free servers across stations
        free: Dict[str, List[Server]] = {}
        for st_id, st in self.stations.items():
            for i, srv in enumerate(st.servers):
                if not srv.busy:
                    free.setdefault(st.station_id, []).append(srv)

        if not free:
            return

        assignments = self.policy.decide(self, self.t, free_servers=free)

        # start service for each assigned server
        for srv, q in assignments.items():
            if len(q) == 0 or srv.busy:
                continue
            job = q.pop()
            # start service
            dep_time = srv.start_service(job, self.t)
            # schedule departure
            st_id = q.station_id  # job currently at that station
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
        return self.sum_sojourn / self.completed_jobs if self.completed_jobs else float("nan")


# -------- Utility samplers (Exp arrivals/services) --------

def exp_interarrival(rate: float, rng: np.random.Generator) -> Callable[[], float]:
    assert rate > 0
    return lambda: rng.exponential(1.0 / rate)

def exp_service(mu: float, rng: np.random.Generator) -> Callable[[Job], float]:
    assert mu > 0
    return lambda job: rng.exponential(1.0 / mu)


# In[47]:


# ----------------- CrissCross subclass -----------------

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional

class CrissCross(Network):
    """
      - External class 1 -> S1:Q1
      - External class 2 -> S1:Q2
      - After S1 service: becomes class 3 -> S2:Q1
      - After S2 service: exit
    Stations:
      S1 has queues Q1 (cls '1') and Q2 (cls '2'), c1 servers.
      S2 has queue  Q1 (cls '3'), c2 servers.
    """

    def __init__(self,
                 policy: SchedulingPolicy,
                 *,
                 seed: int = 0,
                 c1: int = 1, c2: int = 1,
                 lam1: float = 0.8, lam2: float = 0.8,
                 mu1: float = 2.4,  # S1 service rate for class '1'
                 mu2: float = 2.4,  # S1 service rate for class '2'
                 mu3: float = 2.4   # S2 service rate for class '3'
                 ):
        rng = np.random.default_rng(seed)

        # --- service samplers exactly as you specified ---
        def svc_S1(job: Job) -> float:
            rate = mu1 if job.cls == "1" else mu2   # S1 serves '1' or '2'
            return rng.exponential(1.0 / rate)

        def svc_S2(job: Job) -> float:
            # S2 serves only class '3'
            rate = mu3
            return rng.exponential(1.0 / rate)

        # --- stations/queues ---
        S1 = Station(
            "S1",
            queues={"Q1": Queue("S1", "Q1"),   # class '1'
                    "Q2": Queue("S1", "Q2")},  # class '2'
            servers=[Server(f"S1-s{i}", svc_S1) for i in range(c1)]
        )
        S2 = Station(
            "S2",
            queues={"Q1": Queue("S2", "Q1")},  # class '3'
            servers=[Server(f"S2-s{i}", svc_S2) for i in range(c2)]
        )

        # --- external arrivals (both to S1) ---
        ap1 = ArrivalProcess("S1", "Q1", lambda: rng.exponential(1.0 / lam1), job_class="1")
        ap2 = ArrivalProcess("S1", "Q2", lambda: rng.exponential(1.0 / lam2), job_class="2")

        # --- router: after S1 -> S2 as class '3'; after S2 -> exit ---
        class _Router():
            def route(self, job: Job, station_id: str, t: float) -> Optional[Tuple[str, str]]:
                if station_id == "S1":
                    job.cls = "3"
                    return ("S2", "Q1")
                if station_id == "S2":
                    return None
                return None

        super().__init__(
            stations={"S1": S1, "S2": S2},
            arrivals=[ap1, ap2],
            router=_Router(),
            policy=policy,
            rng=rng,
        )

        # useful for MaxPressure (pressure uses downstream queue length)
        self._next_hop = {
            ("S1", "Q1"): ("S2", "Q1"),
            ("S1", "Q2"): ("S2", "Q1"),
            ("S2", "Q1"): None,
        }

        # keep parameters (handy for titles/diagnostics)
        self._params = dict(c1=c1, c2=c2, lam1=lam1, lam2=lam2, mu1=mu1, mu2=mu2, mu3=mu3)

    # ----------------- Metrics helpers (scoped to this topology) -----------------

    def reset_metrics(self) -> None:
        self.completed_jobs = 0
        self.sum_sojourn = 0.0
        if hasattr(self, "exited_jobs"):
            self.exited_jobs.clear()
        self._measure_t0 = self.t 
        
    def summarize(self) -> Dict[str, Any]:
        jobs = getattr(self, "exited_jobs", [])
        window = max(1e-12, self.t - getattr(self, "_measure_t0", 0.0))  # avoid /0
        if not jobs:
            return {
                "completed": self.completed_jobs,
                "mean_sojourn": self.mean_sojourn(),
                "p50": float("nan"),
                "p90": float("nan"),
                "p95": float("nan"),
                "mean_1": float("nan"),
                "mean_2": float("nan"),
                "mean_3": float("nan"),
                "throughput_per_time": float("nan"),
            }
        soj = np.array([j.t_departure - j.t_arrival for j in jobs], dtype=float)
        cls1 = np.array([j.cls == "1" for j in jobs])
        cls2 = np.array([j.cls == "2" for j in jobs])
        cls3 = np.array([j.cls == "3" for j in jobs])

        def m(mask): return float(soj[mask].mean()) if mask.any() else float("nan")
        def pct(a, p): return float(np.percentile(a, p)) if a.size else float("nan")
        return {
            "completed": int(len(jobs)),
            "mean_sojourn": float(soj.mean()),
            "p50": pct(soj, 50), "p90": pct(soj, 90), "p95": pct(soj, 95),
            "mean_1": m(cls1), "mean_2": m(cls2), "mean_3": m(cls3),
            "throughput_per_time": float(len(jobs) / window)
        }

    def run_warmup_and_measure(self, warmup_time: float = 1_000.0, measure_time: float = 2_000.0) -> Dict[str, Any]:
        self.run(until_time=warmup_time)
        self.reset_metrics()
        t0 = self.t
        self.run(until_time=t0 + measure_time)
        return self.summarize()

    # ----------------- Stability helpers for YOUR topology -----------------

    @staticmethod
    def capacity_ok(lam1: float, lam2: float, c1: int, c2: int, mu1: float, mu2: float, mu3: float) -> Dict[str, float]:
        """
        M/M/c station loads for this topology.
        Station 1: mix of classes 1 & 2 -> ρ1 = (λ1/μ1 + λ2/μ2) / c1
        Station 2: all jobs go through as class 3 -> ρ2 = (λ1 + λ2) / (c2 * μ3)
        """
        rho1 = (lam1 / mu1 + lam2 / mu2) / c1
        rho2 = (lam1 + lam2) / (c2 * mu3)
        return {"rho1": rho1, "rho2": rho2, "stable": float(rho1 < 1 and rho2 < 1)}

    @staticmethod
    def lambda_star_symmetric(mu1: float, mu2: float, mu3: float, c1: int = 1, c2: int = 1) -> float:
        """
        Max symmetric λ (λ1=λ2=λ) keeping both stations stable.
        From ρ1<1: 2λ*(0.5*(1/μ1+1/μ2)) / c1 < 1  => λ < c1 / (1/μ1 + 1/μ2)
        From ρ2<1: (2λ)/(c2 μ3) < 1               => λ < (c2 μ3)/2
        """
        return min(c1 / (1.0 / mu1 + 1.0 / mu2), (c2 * mu3) / 2.0)

    # ----------------- Policy experiments (run/plot) -----------------

    @classmethod
    def run_policy_at_lambda(cls,
                             policy_name: str,
                             policy: SchedulingPolicy,
                             lam: float,
                             *,
                             reps: int = 5,
                             warmup_time: float = 1_000.0,
                             measure_time: float = 2_000.0,
                             seed0: int = 123,
                             c1: int = 1, c2: int = 1,
                             mu1: float = 2.4, mu2: float = 2.4, mu3: float = 2.4
                             ) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        for r in range(reps):
            net = cls(policy=policy, seed=seed0 + r,
                      c1=c1, c2=c2, lam1=lam, lam2=lam, mu1=mu1, mu2=mu2, mu3=mu3)
            m = net.run_warmup_and_measure(warmup_time=warmup_time, measure_time=measure_time)
            m["policy"] = policy_name
            m["lam"] = lam
            rows.append(m)

        def agg(key: str) -> Tuple[float, float, int]:
            vals = np.array([row.get(key, np.nan) for row in rows], dtype=float)
            good = ~np.isnan(vals)
            if not np.any(good): return (float("nan"), 0.0, 0)
            return (float(np.nanmean(vals)), float(np.nanstd(vals, ddof=0)), int(np.sum(good)))

        W_mean, W_sd, _  = agg("mean_sojourn")
        p95_mean, _, _   = agg("p95")
        thr_mean, _, _   = agg("throughput_per_time")
        return {"policy": policy_name, "lam": lam, "W_mean": W_mean, "W_sd": W_sd,
                "p95_mean": p95_mean, "throughput": thr_mean, "rows": rows}

    @classmethod
    def sweep_and_plot(cls,
                       policies: List[Tuple[str, SchedulingPolicy]],
                       lam_grid: np.ndarray,
                       *,
                       reps: int = 5,
                       warmup_time: float = 1_000.0,
                       measure_time: float = 5_000.0,
                       seed0: int = 777,
                       c1: int = 1, c2: int = 1,
                       mu1: float = 2.4, mu2: float = 2.4, mu3: float = 2.4,
                       title: str = "Criss-Cross (yours): Mean Sojourn vs λ (λ₁=λ₂=λ)"
                       ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for lam in lam_grid:
            for name, pol in policies:
                res = cls.run_policy_at_lambda(name, pol, lam,
                                               reps=reps, warmup_time=warmup_time, measure_time=measure_time,
                                               seed0=seed0, c1=c1, c2=c2, mu1=mu1, mu2=mu2, mu3=mu3)
                results.append(res)

        # organize & plot
        by_policy: Dict[str, Dict[str, np.ndarray]] = {}
        for r in results:
            d = by_policy.setdefault(r["policy"], {"lam": [], "W_mean": [], "W_sd": []})
            d["lam"].append(r["lam"]); d["W_mean"].append(r["W_mean"]); d["W_sd"].append(r["W_sd"])
        for pol, d in by_policy.items():
            order = np.argsort(np.array(d["lam"]))
            d["lam"]   = np.array(d["lam"])[order]
            d["W_mean"]= np.array(d["W_mean"])[order]
            d["W_sd"]  = np.array(d["W_sd"])[order]

        plt.figure(figsize=(7.5, 4.5))
        for pol, d in by_policy.items():
            plt.errorbar(d["lam"], d["W_mean"], yerr=d["W_sd"], marker='o',
                         linewidth=1.5, capsize=3, label=pol)
        lam_star = cls.lambda_star_symmetric(mu1, mu2, mu3, c1=c1, c2=c2)
        plt.axvline(lam_star, linestyle='--', linewidth=1.0)
        plt.title(title)
        plt.xlabel("Arrival rate λ (λ₁=λ₂=λ)")
        plt.ylabel("Mean sojourn time W")
        plt.legend()
        plt.tight_layout()
        plt.show()
        return results

    # ------------ convenience for MaxPressure on YOUR topology -----------

    @property
    def next_hop(self) -> Dict[Tuple[str, str], Optional[Tuple[str, str]]]:
        return dict(self._next_hop)

    def make_maxpressure_policy(self) -> "MaxPressurePolicy":
        if "MaxPressurePolicy" not in globals():
            raise AttributeError("MaxPressurePolicy not defined.")
        return MaxPressurePolicy(self.next_hop)


# In[62]:


# ----------------- KellyNetwork subclass -----------------

class KellyNetwork(Network):
    """
      - External class 1 -> S1:Q1
      - External class 2 -> S1:Q2
      - After S1 service:
          * class 1 becomes class 3 -> S2:Q1 -> exit
          * class 2 exits
    Stations:
      S1 has queues Q1 (cls '1') and Q2 (cls '2'), c1 servers.
      S2 has queue  Q1 (cls '3'), c2 servers.
    """

    def __init__(self,
                 policy: SchedulingPolicy,
                 *,
                 seed: int = 0,
                 c1: int = 1, c2: int = 1,
                 lam1: float = 0.8, lam2: float = 0.8,      # externals
                 mu1: float = 2.4,  # S1 service rate for class '1'
                 mu2: float = 1.6,  # S1 service rate for class '2'
                 mu3: float = 2.0   # S2 service rate for class '3'
                 ):
        rng = np.random.default_rng(seed)

        # --- service samplers (per station; class-based at S1) ---
        def svc_S1(job: Job) -> float:
            rate = mu1 if job.cls == "1" else mu2   # S1 serves '1' and '2'
            return rng.exponential(1.0 / rate)

        def svc_S2(job: Job) -> float:
            # S2 serves only '3'
            return rng.exponential(1.0 / mu3)

        # --- stations/queues ---
        S1 = Station(
            "S1",
            queues={"Q1": Queue("S1", "Q1"),   # holds class '1'
                    "Q2": Queue("S1", "Q2")},  # holds class '2'
            servers=[Server(f"S1-s{i}", svc_S1) for i in range(c1)]
        )
        S2 = Station(
            "S2",
            queues={"Q1": Queue("S2", "Q1")},  # holds class '3'
            servers=[Server(f"S2-s{i}", svc_S2) for i in range(c2)]
        )

        # --- external arrivals (both into S1) ---
        ap1 = ArrivalProcess("S1", "Q1", lambda: rng.exponential(1.0 / lam1), job_class="1")
        ap2 = ArrivalProcess("S1", "Q2", lambda: rng.exponential(1.0 / lam2), job_class="2")

        # --- router: after S1, 1 -> S2 as 3; 2 -> exit; after S2 -> exit ---
        class _Router():
            def route(self, job: Job, station_id: str, t: float) -> Optional[Tuple[str, str]]:
                if station_id == "S1":
                    if job.cls == "1":
                        job.cls = "3"
                        return ("S2", "Q1")
                    if job.cls == "2":
                        return None
                if station_id == "S2":
                    return None
                return None

        super().__init__(
            stations={"S1": S1, "S2": S2},
            arrivals=[ap1, ap2],
            router=_Router(),
            policy=policy,
            rng=rng,
        )

        # for MaxPressure: (S1:Q1 -> S2:Q1), (S1:Q2 -> exit), (S2:Q1 -> exit)
        self._next_hop = {
            ("S1", "Q1"): ("S2", "Q1"),
            ("S1", "Q2"): None,
            ("S2", "Q1"): None,
        }

        # keep parameters handy
        self._params = dict(c1=c1, c2=c2, lam1=lam1, lam2=lam2, mu1=mu1, mu2=mu2, mu3=mu3)

    # ----------------- Metrics helpers -----------------

    def reset_metrics(self) -> None:
        self.completed_jobs = 0
        self.sum_sojourn = 0.0
        if hasattr(self, "exited_jobs"):
            self.exited_jobs.clear()
        self._measure_t0 = self.t

    def summarize(self) -> Dict[str, Any]:
        jobs = getattr(self, "exited_jobs", [])
        window = max(1e-12, self.t - getattr(self, "_measure_t0", 0.0))
        if not jobs:
            return {
                "completed": self.completed_jobs,
                "mean_sojourn": self.mean_sojourn(),
                "p50": float("nan"),
                "p90": float("nan"),
                "p95": float("nan"),
                "mean_1": float("nan"),
                "mean_2": float("nan"),
                "mean_3": float("nan"),
                "throughput_per_time": float("nan"),
            }
        soj = np.array([j.t_departure - j.t_arrival for j in jobs], dtype=float)
        cls1 = np.array([j.cls == "1" for j in jobs])
        #False...
        cls2 = np.array([j.cls == "2" for j in jobs])
        cls3 = np.array([j.cls == "3" for j in jobs])

        def m(mask): return float(soj[mask].mean()) if mask.any() else float("nan")
        def pct(a, p): return float(np.percentile(a, p)) if a.size else float("nan")
        return {
            "completed": int(len(jobs)),
            "mean_sojourn": float(soj.mean()),
            "p50": pct(soj, 50), "p90": pct(soj, 90), "p95": pct(soj, 95),
            "mean_1": m(cls1), "mean_2": m(cls2), "mean_3": m(cls3),
            "throughput_per_time": float(len(jobs) / window),
        }

    def run_warmup_and_measure(self,
                               warmup_time: float = 1_000.0,
                               measure_time: float = 2_000.0) -> Dict[str, Any]:
        self.run(until_time=warmup_time)
        self.reset_metrics()
        t0 = self.t
        self.run(until_time=t0 + measure_time)
        return self.summarize()

    # ----------------- Stability helpers -----------------
    @staticmethod
    def capacity_ok(lam1: float, lam2: float, c1: int, c2: int,
                    mu1: float, mu2: float, mu3: float) -> Dict[str, float]:
        """
        Station 1 load: ρ1 = (λ1/μ1 + λ2/μ2) / c1
        Station 2 load: ρ2 = (λ1/μ3) / c2    (only stream 1 continues)
        """
        rho1 = (lam1 / mu1 + lam2 / mu2) / c1
        rho2 = (lam1 / mu3) / c2
        return {"rho1": rho1, "rho2": rho2, "stable": float(rho1 < 1 and rho2 < 1)}

    @staticmethod
    def lambda_star_symmetric(mu1: float, mu2: float, mu3: float,
                              c1: int = 1, c2: int = 1) -> float:
        """
        If λ1=λ2=λ, stability requires:
          ρ1 = (λ/μ1 + λ/μ2)/c1  < 1  =>  λ < c1 / (1/μ1 + 1/μ2)
          ρ2 = (λ/μ3)/c2         < 1  =>  λ < c2 * μ3
        """
        return min(c1 / (1.0/mu1 + 1.0/mu2), c2 * mu3)

    # ----------------- Policy experiments (same pattern) -----------------
    @classmethod
    def run_policy_at_lambda(cls,
                             policy_name: str,
                             policy: SchedulingPolicy,
                             lam: float,
                             *,
                             reps: int = 5,
                             warmup_time: float = 1_000.0,
                             measure_time: float = 2_000.0,
                             seed0: int = 123,
                             c1: int = 1, c2: int = 1,
                             mu1: float = 2.4, mu2: float = 1.6, mu3: float = 2.0
                             ) -> Dict[str, Any]:
        """
        Build fresh Kelly networks with λ1=λ2=lam for each replication, run warm-up + measure,
        and aggregate mean sojourn, p95, and throughput.
        """
        rows: List[Dict[str, Any]] = []
        for r in range(reps):
            net = cls(policy=policy, seed=seed0 + r,
                      c1=c1, c2=c2,
                      lam1=lam, lam2=lam,
                      mu1=mu1, mu2=mu2, mu3=mu3)
            m = net.run_warmup_and_measure(warmup_time=warmup_time, measure_time=measure_time)
            m["policy"] = policy_name
            m["lam"] = lam
            rows.append(m)

        def agg(key: str) -> Tuple[float, float, int]:
            vals = np.array([row.get(key, np.nan) for row in rows], dtype=float)
            good = ~np.isnan(vals)
            if not np.any(good): return (float("nan"), 0.0, 0)
            return (float(np.nanmean(vals)), float(np.nanstd(vals, ddof=0)), int(np.sum(good)))

        W_mean, W_sd, _  = agg("mean_sojourn")
        p95_mean, _, _   = agg("p95")
        thr_mean, _, _   = agg("throughput_per_time")

        return {
            "policy": policy_name,
            "lam": lam,
            "W_mean": W_mean,
            "W_sd": W_sd,
            "p95_mean": p95_mean,
            "throughput": thr_mean,
            "rows": rows,
        }

    @classmethod
    def sweep_and_plot(cls,
                       policies: List[Tuple[str, SchedulingPolicy]],
                       lam_grid: np.ndarray,
                       *,
                       reps: int = 5,
                       warmup_time: float = 1_000.0,
                       measure_time: float = 5_000.0,
                       seed0: int = 777,
                       c1: int = 1, c2: int = 1,
                       mu1: float = 2.4, mu2: float = 1.6, mu3: float = 2.0,
                       title: str = "Kelly Network: Mean Sojourn vs λ (λ₁=λ₂=λ)"
                       ) -> List[Dict[str, Any]]:
        """
        Sweep symmetric λ over lam_grid for the given policies, plot mean sojourn with error bars,
        and return the aggregated results.
        """
        results: List[Dict[str, Any]] = []
        for lam in lam_grid:
            for name, pol in policies:
                res = cls.run_policy_at_lambda(name, pol, lam,
                                               reps=reps,
                                               warmup_time=warmup_time,
                                               measure_time=measure_time,
                                               seed0=seed0,
                                               c1=c1, c2=c2,
                                               mu1=mu1, mu2=mu2, mu3=mu3)
                results.append(res)

        # organize for plotting
        by_policy: Dict[str, Dict[str, np.ndarray]] = {}
        for r in results:
            d = by_policy.setdefault(r["policy"], {"lam": [], "W_mean": [], "W_sd": []})
            d["lam"].append(r["lam"])
            d["W_mean"].append(r["W_mean"])
            d["W_sd"].append(r["W_sd"])

        for pol, d in by_policy.items():
            order = np.argsort(np.array(d["lam"]))
            d["lam"] = np.array(d["lam"])[order]
            d["W_mean"] = np.array(d["W_mean"])[order]
            d["W_sd"] = np.array(d["W_sd"])[order]

        plt.figure(figsize=(7.5, 4.5))
        for pol, d in by_policy.items():
            plt.errorbar(d["lam"], d["W_mean"], yerr=d["W_sd"],
                         marker='o', linewidth=1.5, capsize=3, label=pol)

        # show symmetric capacity limit (min of station constraints)
        lam_star = cls.lambda_star_symmetric(mu1, mu2, mu3, c1=c1, c2=c2)
        plt.axvline(lam_star, linestyle='--', linewidth=1.0)
        plt.title(title)
        plt.xlabel("Arrival rate λ (per external class)")
        plt.ylabel("Mean sojourn time W")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return results


# In[1]:


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
        rng = np.random.default_rng(seed)
        lam = 9.0 / 140.0
        mu_rates = { 
            1: 1.0 / 8.0, 2: 1.0 / 2.0, 3: 1.0 / 4.0,
            4: 1.0 / 6.0, 5: 1.0 / 7.0, 6: 1.0 / 1.0,
        }

        def service_sampler(job: Job) -> float:
            class_idx = int(job.cls)
            key = (class_idx - 1) % 6 + 1
            rate = mu_rates[key]
            return rng.exponential(1.0 / rate)

        stations: Dict[str, Station] = {}
        for i in range(1, L + 1):
            sid = f"S{i}"
            station_queues: Dict[str, Queue] = {}
            for k in range(1, 4):
                class_id = 3 * (i - 1) + k
                qid = f"Q{class_id}"
                station_queues[qid] = Queue(sid, qid)
            station_servers = [Server(f"{sid}-s0", service_sampler)]
            stations[sid] = Station(sid, station_queues, station_servers)

        arrival_sampler = lambda: rng.exponential(1.0 / lam)
        ap1 = ArrivalProcess("S1", "Q1", arrival_sampler, job_class="1")
        ap3 = ArrivalProcess("S1", "Q3", arrival_sampler, job_class="3")

        class _Router:
            def route(self, job: Job, station_id: str, t: float) -> Optional[Tuple[str, str]]:
                station_num = int(station_id.replace("S", ""))
                class_num = int(job.cls)
                if station_num < L:
                    next_class = class_num + 3
                    next_station = station_num + 1
                    job.cls = str(next_class)
                    return (f"S{next_station}", f"Q{next_class}")
                elif station_num == L:
                    if class_num == 3 * (L - 1) + 1:
                        job.cls = "2"
                        return ("S1", "Q2")
                    else:
                        return None
                return None

        super().__init__(
            stations=stations,
            arrivals=[ap1, ap3],
            router=_Router(),
            policy=policy,
            rng=rng,
        )
        self._params = dict(L=L, lam=lam, mu_rates=mu_rates)

    # --- Metrics and Experiment Helpers ---

    def run_and_get_batch_means_stats(
        self,
        warmup_time: float,
        num_batches: int,
        batch_duration: float
    ) -> Dict[str, Any]:
        """
        Runs a simulation with warmup and uses the batch means method to
        get a stable estimate of the mean number of jobs in the system.
        """
        print(f"Running warmup for {warmup_time:.0f} time units...")
        self.run(until_time=warmup_time)
        print("Warmup complete. Starting batch means measurement...")

        batch_means = []
        
        for i in range(num_batches):
            # Reset the area counters for queues and servers at the start of the batch
            for st in self.stations.values():
                if hasattr(st, "_ql_area"):
                    st._ql_area = {qid: 0.0 for qid in st.queues}
                if hasattr(st, "_sl_area"): # NEW
                    st._sl_area = 0.0       # NEW

            t_batch_start = self.t
            self.run(until_time=t_batch_start + batch_duration)
            
            # Calculate the mean jobs for this batch (queues + servers)
            total_area_this_batch = 0
            for st in self.stations.values():
                if hasattr(st, "_ql_area"):
                    total_area_this_batch += sum(st._ql_area.values())
                if hasattr(st, "_sl_area"): # NEW
                    total_area_this_batch += st._sl_area # NEW
            
            mean_jobs_this_batch = total_area_this_batch / batch_duration
            batch_means.append(mean_jobs_this_batch)
            
            # Optional: uncomment to see progress
            # print(f"Batch {i+1}/{num_batches} complete. Mean jobs: {mean_jobs_this_batch:.3f}")

        # Calculate statistics over the batch means
        mean_of_means = np.mean(batch_means)
        std_of_means = np.std(batch_means, ddof=1)
        
        # 95% CI half-width using z=1.96 (appropriate for num_batches >= 30)
        ci_half_width = 1.96 * (std_of_means / np.sqrt(num_batches))

        print("Measurement complete.")
        return {
            "mean_jobs_in_system": mean_of_means,
            "ci_half_width": ci_half_width,
            "std_dev_of_batch_means": std_of_means,
            "num_batches": num_batches,
        }


# In[ ]:




