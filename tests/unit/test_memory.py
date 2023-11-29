import psutil
import time


def test_memory_bound():
    print(psutil.virtual_memory())
    alloc = []
    alloc_unit = bytearray(1024 * 1024 * 100)
    while True:
        alloc.append(alloc_unit)
        print(psutil.virtual_memory())
        time.sleep(5)
