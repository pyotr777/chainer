import os
import logging

debug = True
time_function_node = False
time_cuda = False
time_optimizer_update = False
time_convert = False
log_convolution = False  # Log timings in convolution_2d.py
log_convolution_forward = True
log_convolution_backward = True
log_convolution_backward_data = True

wd = os.getcwd()
logdir = os.path.join(wd,"timelogs")

import string

# Replaces "," with ":"
def csvValue(s,bad=",",good=":"):
    s = str(s)
    s = string.replace(s,bad,good)
    return s


def LogFile(logfile):
    logging.basicConfig(filename=logfile,level=logging.DEBUG)
    print("Logging to {}".format(logfile))
    
    if time_function_node:
        logging.debug("time1;time2;time3;Input;Class")
    else:
        logging.debug("Address;Parameter;Value")
    return