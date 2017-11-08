#!/usr/bin/env python3
"""
Benchmark various Priority Replay Buffer variants
"""
import timeit
import numpy as np
import collections

SIZES = [10**n for n in (3, 4, 5)]
DATA_SHAPE = (84, 84, 4)
REPEAT_NUMBER = 10


class PrioReplayBufferDeque:
    def __init__(self, buf_size, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.buffer = collections.deque(maxlen=buf_size)
        self.priorities = collections.deque(maxlen=buf_size)

    def __len__(self):
        return len(self.buffer)

    def append(self, sample):
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(sample)
        self.priorities.append(max_prio)

    def sample(self, batch_size, beta=0.4):
        probs = np.array(self.priorities, dtype=np.float32) ** self.prob_alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


class PrioReplayBufferList:
    def __init__(self, buf_size, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def append(self, sample):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = np.array(prios, dtype=np.float32) ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


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
    bench_buffer(PrioReplayBufferList)
    bench_buffer(PrioReplayBufferDeque)
    pass
