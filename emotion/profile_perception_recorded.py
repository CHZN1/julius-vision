from scripts.experiments.julius_benchmark_paths import BENCHMARK_DIR
"""Controlled face-present workload using a recorded enrollment image, not a live camera."""
import argparse, json, resource, time
from pathlib import Path
import cv2, torch
from emotion.julius_perception import Perception
p=argparse.ArgumentParser();p.add_argument('--fps',type=float,default=8);p.add_argument('--emotion-interval',type=float,default=0);p.add_argument('--minimal',action='store_true');a=p.parse_args()
image=cv2.imread('/private/tmp/julius_profile_face.jpg');image=cv2.resize(image,(640,480))
path='/private/tmp/julius_profile_face.mp4';writer=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'),8,(640,480))
for _ in range(240):writer.write(image)
writer.release()
t=time.perf_counter(); eyes=Perception(target_fps=a.fps,emotion_interval=a.emotion_interval,camera_source=path,allowed_modules=['detection','recognition'] if a.minimal else None).start();load=time.perf_counter()-t
r0=resource.getrusage(resource.RUSAGE_SELF);t=time.perf_counter();time.sleep(20);r1=resource.getrusage(resource.RUSAGE_SELF);seconds=time.perf_counter()-t
d={'minimal_modules':a.minimal,'input':'repeated recorded face image','fps':a.fps,'emotion_interval':a.emotion_interval,'load_seconds':load,'sample_seconds':seconds,'cpu_seconds':r1.ru_utime+r1.ru_stime-r0.ru_utime-r0.ru_stime,'peak_rss_bytes':r1.ru_maxrss,'state':eyes.snapshot(),'camera_available':eyes.camera_available,'mps_active_bytes':torch.mps.current_allocated_memory(),'mps_driver_bytes':torch.mps.driver_allocated_memory()}
eyes.stop();print(json.dumps(d),flush=True);(BENCHMARK_DIR/f'benchmark_perception_recorded_{a.fps:g}_{a.emotion_interval:g}{'_minimal' if a.minimal else ''}.json').write_text(json.dumps(d,indent=2))
