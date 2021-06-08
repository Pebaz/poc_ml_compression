from threading import Thread
from queue import SimpleQueue, Empty


QueueGetItemTimeout = Empty


class PoisonPill(Exception):
    pass


class ExceptionRaised(PoisonPill):
    def __init__(self, exception):
        super().__init__(exception)
        self.exception = exception


class QueueFinished(PoisonPill):
    pass


class QueueKilled(PoisonPill):
    pass


class Query(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.killed = False
        self.queue = SimpleQueue()

    def kill(self, join=False):
        self.killed = True
        if join:
            self.join()

    def run(self):
        try:
            if self._target:
                self._target(self, *self._args, **self._kwargs)
        except Exception as e:
            self.queue.put(ExceptionRaised(e))
        finally:
            del self._target, self._args, self._kwargs

