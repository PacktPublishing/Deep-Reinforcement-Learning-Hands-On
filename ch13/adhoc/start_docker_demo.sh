#!/usr/bin/env bash
set -x
docker run -e TURK_DB='' -p 5899:5899 --privileged --ipc host --cap-add SYS_ADMIN shmuma/miniwob demonstration -e ${1:-wob.mini.ClickTest2-v0}
