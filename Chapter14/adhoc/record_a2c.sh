#!/usr/bin/env bash
xvfb-run -s "-screen 0 640x480x24 +extension GLX" ./03_play_a2c.py -m $1 -r $2
