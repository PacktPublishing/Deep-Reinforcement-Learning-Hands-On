#!/usr/bin/env bash
docker run -d -p 5900:5900 -p 15900:15900 --privileged --ipc host --cap-add SYS_ADMIN 92756d1f08ac
docker run -d -p 5901:5900 -p 15901:15900 --privileged --ipc host --cap-add SYS_ADMIN 92756d1f08ac
docker run -d -p 5902:5900 -p 15902:15900 --privileged --ipc host --cap-add SYS_ADMIN 92756d1f08ac
docker run -d -p 5903:5900 -p 15903:15900 --privileged --ipc host --cap-add SYS_ADMIN 92756d1f08ac
