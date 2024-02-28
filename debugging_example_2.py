import pydevd_pycharm
import torch

pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

print('Cuda available?', torch.cuda.is_available())
print("Done.")
