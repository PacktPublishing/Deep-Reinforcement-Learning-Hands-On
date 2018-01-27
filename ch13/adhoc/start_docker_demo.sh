#!/usr/bin/env bash
set -x
docker run -e TURK_DB='' -p 5910:5900 -p 5899:5899 -p 15910:15900 --privileged --ipc host --cap-add SYS_ADMIN 92756d1f08ac demonstration -e ${1:-wob.mini.ClickTest2-v0}
