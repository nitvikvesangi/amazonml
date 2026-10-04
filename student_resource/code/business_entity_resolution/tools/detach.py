#!/usr/bin/env python3
"""Run a command fully detached (double fork + setsid), logging to a file.

Long jobs must not be children of a tool-call shell: when that shell exits the
whole process group can be signalled. This reparents the job to init.

Usage:
  python3 tools/detach.py logs/job.log python3 -u tools/thing.py --arg 1

Then poll with:  tail -f logs/job.log
"""
import os
import subprocess
import sys

if len(sys.argv) < 3:
    sys.exit("usage: detach.py LOGFILE CMD [ARGS...]")

if os.fork() == 0:
    os.setsid()
    if os.fork() == 0:
        log = open(sys.argv[1], 'a')
        subprocess.Popen(sys.argv[2:], stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL)
    os._exit(0)
os._exit(0)
