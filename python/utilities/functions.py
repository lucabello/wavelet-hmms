import uuid
import subprocess
import time
from math import log


def random_string(length=8):
    random = str(uuid.uuid4()).lower().replace("-", "")
    return random[0:length]


def compute_error(real, measured):
    if abs(real) > abs(measured):
        maximum = abs(real)
    else:
        maximum = abs(measured)
    if real == 0 and measured == 0:
        return 0
    return abs((real - measured) / maximum)


def compute_kl(gaussian1, gaussian2):
    m1, s1 = gaussian1
    m2, s2 = gaussian2
    a = log(s2/s1)
    b = (s1**2 + (m1-m2)**2)/(2*(s2**2))
    return a + b - 0.5


def execute_and_time(args):
    start = time.perf_counter()
    subprocess.call(args)
    end = time.perf_counter()
    return end-start


def state_path_differences(filename1, filename2):
    difference_count = 0
    with open(filename1, "r") as f1:
        with open(filename2, "r") as f2:
            for s1, s2 in zip(f1.read().split(), f2.read().split()):
                if s1 != s2:
                    difference_count += 1
    return difference_count
    return difference_count
