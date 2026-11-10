from scripts.experiments.julius_benchmark_paths import BENCHMARK_DIR
import json, resource, time
from pathlib import Path
import torch
from emotion.julius_perception import Perception
start=time.perf_counter(); p=Perception().start(); loaded=time.perf_counter()-start
r0=resource.getrusage(resource.RUSAGE_SELF)
time.sleep(20)
r1=resource.getrusage(resource.RUSAGE_SELF)
result={'load_seconds':loaded,'sample_seconds':20,'cpu_seconds':r1.ru_utime+r1.ru_stime-r0.ru_utime-r0.ru_stime,'peak_rss_bytes':r1.ru_maxrss,'state':p.snapshot(),'mps_active_bytes':torch.mps.current_allocated_memory(),'mps_driver_bytes':torch.mps.driver_allocated_memory()}
p.stop(); print(json.dumps(result),flush=True); (BENCHMARK_DIR/'benchmark_perception.json').write_text(json.dumps(result,indent=2))
