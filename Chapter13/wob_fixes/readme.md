**UPD**: both patches are applied to the image [shmuma/miniwob:v2](https://hub.docker.com/r/shmuma/miniwob/), so instructions below are left here only if you're curious or want to reproduce those images. In other cases, you can just use patched image from my Docker Hub repo.

# Fixes of WoB docker image

Sometimes WoB docker process crashes with the following stack trace:
```
[2018-01-19 15:51:40,618] [instruction] Sent env text Create a line that bisects the angle evenly in two, then press submit.
[2018-01-19 15:51:40,618] instruction generated 
Namespace(env_id='wob.mini.ClickShape-v0', fps=15, idle_timeout=None, mode='ENV', verbosity=0)
[EnvController] found 80 miniwob envs
Launching new Chrome process...
window.location.replace("http://localhost/miniwob/click-shape.html")
window.location.replace("http://localhost/miniwob/bisect-angle.html")
Traceback (most recent call last):
  File "bin/run.py", line 704, in <module>
    sys.exit(main())
  File "bin/run.py", line 698, in main
    error_buffer.blocking_check(timeout=60)
  File "/app/universe/universe/utils/__init__.py", line 56, in blocking_check
    self.check(timeout)
  File "/app/universe/universe/utils/__init__.py", line 48, in check
    raise error
universe.error.Error: Traceback (most recent call last):
  File "bin/run.py", line 486, in run
    self.do_run()
  File "bin/run.py", line 587, in do_run
    reward = reward_client + reward_server
TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'
```

As there is no open-sourced version of WoB or other Universe docker images,
 the only way to fix this is to change the image directly. Below are the 
 instruction how to do this.

Another issue found a bit later is that reward proxy daemon is constantly overwrites the reward data.
This problem is fixed by the second patch.
 
## Starting the image

Start WoB container by running
```bash
docker run -d -p 5900:5900 -p 15900:15900 --privileged --ipc host --cap-add SYS_ADMIN quay.io/openai/universe.world-of-bits:0.20.0
```

In ``docker ps`` command output you should see your container:
```text
shmuma@gpu:~$ docker ps
CONTAINER ID        IMAGE                                          COMMAND                  CREATED             STATUS              PORTS                                              NAMES
d521098d9c48        quay.io/openai/universe.world-of-bits:0.20.0   "/app/universe-envs/w"   4 seconds ago       Up 2 seconds        0.0.0.0:5900->5900/tcp, 0.0.0.0:15900->15900/tcp   reverent_allen
```

Remember the container ID, we'll use it later. In my case it's d521098d9c48.
 
## Patch the image

Apply the image with provided patch 01_wob_crash-fix.patch by running:
```bash
shmuma@gpu:~/work/rl_book_samples/ch13/wob_fixes$ docker exec -i d521098d9c48 patch -d / /app/universe-envs/world-of-bits/bin/run.py < 01_wob_crash-fix.patch
patching file /app/universe-envs/world-of-bits/bin/run.py
```

Second patch should be applied the same way:
```bash
shmuma@gpu:~/work/rl_book_samples/ch13/wob_fixes$ docker exec -i d521098d9c48 patch -d / /app/universe/universe/rewarder/reward_proxy_server.py < 02_reward_proxy_append_rewards.patch
patching file /app/universe/universe/rewarder/reward_proxy_server.py
```

## Save modified image

Then you need to save the patched image to be used later
```bash
shmuma@gpu:~/work/rl_book_samples/ch13/wob_fixes$ docker commit d521098d9c48
sha256:037a8fb6a286b267178cc21392d8d27ad0681dcc0be26791052cc27aa86551ed
```

The resulting hash is a new image id which could be used to start new containers with modified server side.
It is visible in ``docker images`` command output:
```bash
shmuma@gpu:~/work/rl_book_samples/ch13/wob_fixes$ docker images
REPOSITORY                              TAG                 IMAGE ID            CREATED              SIZE
<none>                                  <none>              037a8fb6a286        About a minute ago   1.855 GB
```

## Use the modified image

Now you can kill existing container with ``docker kill d521098d9c48`` and use new
image to start new containers like this:
```bash
docker run -d -p 5900:5900 -p 15900:15900 --privileged --ipc host --cap-add SYS_ADMIN 037a8fb6a286 
docker run -d -p 5901:5900 -p 15901:15900 --privileged --ipc host --cap-add SYS_ADMIN 037a8fb6a286 
```
