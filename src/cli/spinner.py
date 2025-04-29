import functools
import logging
import re
import sys
import threading
import traceback
from copy import deepcopy
from datetime import timedelta
from itertools import cycle
from pprint import pprint
from time import time, sleep
from typing import Dict, Optional, Union, List, Tuple, Any
from venv import logger


def _spin(msg, start, frames, _stop_spin):
    while not _stop_spin.is_set():
        frame = next(frames)
        sec, fsec = divmod(round(100 * (time() - start)), 100)
        frame += "  ({} : {}.{:02.0f})".format(msg, timedelta(seconds=sec), fsec)
        print('\r', frame, sep='', end='', flush=True)
        sleep(0.2)


def spinner(msg="Elapsed Time"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper_decrorator(*args, **kwargs):
            _stop_spin = threading.Event()
            start = time()
            _spin_thread = threading.Thread(target=_spin, args=(msg, start, cycle(r'-\|/'), _stop_spin))
            _spin_thread.start()
            try:
                value = func(*args, **kwargs)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                stacktrace = traceback.extract_tb(exc_traceback)
                logger.debug(
                    "response {}".format(value))
                logger.debug(sys.exc_info())
                logger.debug(stacktrace)
                raise
            finally:
                stop = time()
                if _spin_thread:
                    _stop_spin.set()
                    _spin_thread.join()
            #     print()
            #     print("=" * 60)
            #     print("Elapsed Time: ")
            #     print("=" * 60)
            #     pprint(stop - start)
            #     print("=" * 60)
            #     print()
            return value

        return wrapper_decrorator

    return decorator