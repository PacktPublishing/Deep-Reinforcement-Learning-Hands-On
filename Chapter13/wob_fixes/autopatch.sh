#!/bin/bash
CONTAINERID=$(docker run -d -p 5900:5900 -p 15900:15900 --privileged --ipc host --cap-add SYS_ADMIN quay.io/openai/universe.world-of-bits:0.20.0)
docker exec -i  patch -d / /app/universe-envs/world-of-bits/bin/run.py < 01_wob_crash-fix.patch
