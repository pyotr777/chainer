import os

debug = True
time_function_node = False
time_cuda = False
time_optimizer_update = False
time_convert = False
log_convolution = False  # Log timings in convolution_2d.py
log_convolution_forward = False
log_convolution_backward = False
log_convolution_backward_data = True

wd = os.getcwd()
logdir = os.path.join(wd,"timelogs")

import string

# Replaces "," with ":"
def csvValue(s,bad=",",good=":"):
    s = str(s)
    s = string.replace(s,bad,good)
    return s