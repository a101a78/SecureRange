import datetime
import os
import threading
from queue import Queue
from threading import Thread

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


class AsyncLogger:
    def __init__(self, file_name='system.pdf', log_folder='log'):
        if not isinstance(file_name, str):
            raise ValueError("file_name must be a string")
        if log_folder is not None and not isinstance(log_folder, str):
            raise ValueError("log_folder must be a string or None")

        log_folder = log_folder or '.'

        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name, ext = os.path.splitext(file_name)
        self.file_name = os.path.join(log_folder, f"{current_time}_{base_name}{ext if ext else '.pdf'}")

        self.queue = Queue()
        self.messages = []
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
                self.messages.append(message)
            self.queue.task_done()

    def log(self, message):
        if not isinstance(message, str):
            raise ValueError("Log message must be a string")

        current_time = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        formatted_message = f"{current_time}\t{message}"
        self.queue.put(formatted_message)

    def _write_to_pdf(self):
        c = canvas.Canvas(self.file_name, pagesize=letter)
        width, height = letter
        margin = 40
        y = height - margin

        for message in self.messages:
            if y < margin:
                c.showPage()
                y = height - margin
            c.drawString(margin, y, message)
            y -= 14

        c.save()

    def close(self):
        self.queue.put(None)
        self.thread.join()
        with self.lock:
            self._write_to_pdf()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Example usage:
# with AsyncLogger([args]) as logger:
#     logger.log('This is a log message.')
