import ctypes as ct
import pyxrt

if __name__ == "__main__":
    print(f"Hello world from {__name__}")

    print(dir(pyxrt))

    # pyxrt = ct.CDLL(name="/opt/xilinx/xrt/python/pyxrt.cpython-312-x86_64-linux-gnu.so")
    # print(type(pyxrt))
    # print(pyxrt)
    print("Goodbye, World!")