#!/usr/bin/env python3
"""Portable Linux wait4 timing when GNU /usr/bin/time is unavailable."""
import json
import os
from pathlib import Path
import sys
import time

output=Path(sys.argv[1])
started=time.monotonic()
pid=os.fork()
if pid==0:
    os.execv(sys.argv[2],sys.argv[2:])
_,status,usage=os.wait4(pid,0)
output.write_text(json.dumps(dict(wall_s=time.monotonic()-started,
    cpu_s=usage.ru_utime+usage.ru_stime,peak_rss_kb=usage.ru_maxrss,
    timing_backend='wait4'),indent=2)+'\n')
raise SystemExit(os.waitstatus_to_exitcode(status) if os.WIFEXITED(status) else 128+os.WTERMSIG(status))
