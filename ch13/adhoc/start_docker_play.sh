#!/usr/bin/env bash
docker run -d -p 5910:5900 -p 15910:15900 --privileged --ipc host --cap-add SYS_ADMIN 92756d1f08ac
