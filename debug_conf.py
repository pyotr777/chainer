debug = False
time_function_node = False
time_cuda = False
time_optimizer_update = False
time_convert = False
log_convolution = False  # Log timings in convolution_2d.py
log_convolution_forward = False
log_convolution_backward = False


import string

# Replaces "," with ":"
def csvValue(s,bad=",",good=":"):
    s = str(s)
    s = string.replace(s,bad,good)
    return s