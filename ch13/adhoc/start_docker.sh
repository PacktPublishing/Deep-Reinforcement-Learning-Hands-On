#!/usr/bin/env bash
docker run -d -p 5900:5900 -p 15900:15900 --privileged --ipc host --cap-add SYS_ADMIN shmuma/miniwob run -f 20
docker run -d -p 5901:5900 -p 15901:15900 --privileged --ipc host --cap-add SYS_ADMIN shmuma/miniwob run -f 20
docker run -d -p 5902:5900 -p 15902:15900 --privileged --ipc host --cap-add SYS_ADMIN shmuma/miniwob run -f 20
docker run -d -p 5903:5900 -p 15903:15900 --privileged --ipc host --cap-add SYS_ADMIN shmuma/miniwob run -f 20
docker run -d -p 5904:5900 -p 15904:15900 --privileged --ipc host --cap-add SYS_ADMIN shmuma/miniwob run -f 20
docker run -d -p 5905:5900 -p 15905:15900 --privileged --ipc host --cap-add SYS_ADMIN shmuma/miniwob run -f 20
docker run -d -p 5906:5900 -p 15906:15900 --privileged --ipc host --cap-add SYS_ADMIN shmuma/miniwob run -f 20
docker run -d -p 5907:5900 -p 15907:15900 --privileged --ipc host --cap-add SYS_ADMIN shmuma/miniwob run -f 20
