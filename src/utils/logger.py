import datetime
import os
import threading
from queue import Queue
from threading import Thread


class AsyncLogger:
    def __init__(self, file_name='system.log', log_folder='log'):
        if not isinstance(file_name, str):
            raise ValueError("file_name must be a string")
        if log_folder is not None and not isinstance(log_folder, str):
            raise ValueError("log_folder must be a string or None")

        log_folder = log_folder or '.'

        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name, ext = os.path.splitext(file_name)
        self.file_name = os.path.join(log_folder, f"{current_time}_{base_name}{ext}")

        self.queue = Queue()
        self.file = open(self.file_name, 'w')
        self.thread = Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()
        self.lock = threading.Lock()

    def _process_queue(self):
        while True:
            message = self.queue.get()
            if message is None:
                break
            with self.lock:
                self.file.write(message + '\n')
                self.file.flush()
            self.queue.task_done()

    def log(self, message):
        if not isinstance(message, str):
            raise ValueError("Log message must be a string")

        current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        formatted_message = f"{current_time}\t{message}"
        self.queue.put(formatted_message)

    def close(self):
        self.queue.put(None)
        self.thread.join()
        with self.lock:
            if self.file:
                self.file.close()
                self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Example usage:
# with AsyncLogger([args]) as logger:
#     logger.log('This is a log message.')
