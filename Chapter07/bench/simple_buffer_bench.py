#!/usr/bin/env python3
"""
Benchmark various Replay Buffer variants
"""
import timeit
import numpy as np
import collections


SIZES = [10**n for n in (3, 4, 5)]
DATA_SHAPE = (84, 84, 4)
REPEAT_NUMBER = 10


class ExperienceBufferDeque:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[idx] for idx in indices]


class ExperienceBufferCircularList:
    def __init__(self, capacity):
        self.buffer = list()
        self.capacity = capacity
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[idx] for idx in indices]



def fill_buf(buf, size):
    for _ in range(size):
        buf.append(np.zeros(DATA_SHAPE, dtype=np.uint8))


def bench_buffer(buf_class):
    print("Benchmarking %s" % buf_class.__name__)

    for size in SIZES:
        print("  Test size %d" % size)
        ns = globals()
        ns.update(locals())
        t = timeit.timeit('fill_buf(buf, size)', setup='buf = buf_class(size)', number=REPEAT_NUMBER, globals=ns)
        print("  * Initial fill:\t%.2f items/s" % (size*REPEAT_NUMBER / t))
        buf = buf_class(size)
        fill_buf(buf, size)
        ns.update(locals())
        t = timeit.timeit('fill_buf(buf, size)', number=REPEAT_NUMBER, globals=ns)
        print("  * Append:\t\t%.2f items/s" % (size*REPEAT_NUMBER / t))
        t = timeit.timeit('buf.sample(4)', number=REPEAT_NUMBER*100, globals=ns)
        print("  * Sample 4:\t\t%.2f items/s" % (REPEAT_NUMBER*100 / t))
        t = timeit.timeit('buf.sample(8)', number=REPEAT_NUMBER*100, globals=ns)
        print("  * Sample 8:\t\t%.2f items/s" % (REPEAT_NUMBER*100 / t))
        t = timeit.timeit('buf.sample(16)', number=REPEAT_NUMBER*100, globals=ns)
        print("  * Sample 16:\t\t%.2f items/s" % (REPEAT_NUMBER*100 / t))
        t = timeit.timeit('buf.sample(32)', number=REPEAT_NUMBER*100, globals=ns)
        print("  * Sample 32:\t\t%.2f items/s" % (REPEAT_NUMBER*100 / t))



if __name__ == "__main__":
    bench_buffer(ExperienceBufferCircularList)
    bench_buffer(ExperienceBufferDeque)
    pass
