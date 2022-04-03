#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Multiprocessing helpers."""

import multiprocessing as mp
# from torch import multiprocessing as mp
import traceback

from pycls.utils.error_handler import ErrorHandler

import pycls.utils.distributed as du


def run_swa(proc_rank, world_size, error_queue, fun, fun_args, fun_kwargs):
    """Runs a function from a child process."""
    try:
        # Initialize the process group
        print("----------------------------")
        print("Proc_rank: {}, World_size: {}".format(proc_rank, world_size))
        # print(len(fun_args))
        print("----------------------------")

        if len(fun_args)==2:#for testing mode
            cfg=fun_args[-1]
        # elif len(fun_args)==8: 
        #     cfg=fun_args[-1]
        else:#for distrib SWA mode
            cfg = fun_args[-1]
        
        du.init_process_group(cfg, proc_rank, world_size)
        # Run the function
        fun(*fun_args, **fun_kwargs)
    except KeyboardInterrupt:
        # Killed by the parent process
        pass
    except Exception:
        # Propagate exception to the parent process
        error_queue.put(traceback.format_exc())
    finally:
        # Destroy the process group
        print("=================================")
        print("==== TRYING TO DESTROY PROCESS GROUP ====")
        du.destroy_process_group()        
        print("==== FINALLY DESTROYED PROCESS GROUP ====")
        print("=================================")

def run(proc_rank, world_size, error_queue, fun, fun_args, fun_kwargs):
    """Runs a function from a child process."""
    try:
        # Initialize the process group
        print("----------------------------")
        print("Proc_rank: {}, World_size: {}".format(proc_rank, world_size))
        # print(len(fun_args))
        print("----------------------------")

        if len(fun_args)==2:#for testing mode
            cfg=fun_args[-1]
        # elif len(fun_args)==8: #for distrib SWA mode
        #     cfg=fun_args[-1]
        else:
            cfg = fun_args[-3]
        
        du.init_process_group(cfg, proc_rank, world_size)
        # Run the function
        fun(*fun_args, **fun_kwargs)
    except KeyboardInterrupt:
        # Killed by the parent process
        pass
    except Exception:
        # Propagate exception to the parent process
        error_queue.put(traceback.format_exc())
    finally:
        # Destroy the process group
        print("=================================")
        print("==== TRYING TO DESTROY PROCESS GROUP ====")
        du.destroy_process_group()        
        print("==== FINALLY DESTROYED PROCESS GROUP ====")
        print("=================================")


# def multi_proc_run(num_proc, fun, fun_args=(), fun_kwargs={}):#,args=None):
#     """Runs a function in a multi-proc setting."""

#     # Handle errors from training subprocesses
#     error_queue = mp.SimpleQueue()
#     error_handler = ErrorHandler(error_queue)

#     # Run each training subprocess
#     ps = []
#     for i in range(num_proc):
#         p_i = mp.Process(
#             target=run,
#             args=(i, num_proc, error_queue, fun, fun_args, fun_kwargs)
#         )
#         ps.append(p_i)
#         p_i.start()
#         error_handler.add_child(p_i.pid)

#     # Wait for each subprocess to finish
#     for p in ps:
#         p.join()
def swa_multi_proc_run(num_proc, fun, fun_args=(), fun_kwargs={},args=None):
    """Runs a function in a multi-proc setting for SWA"""

    # Handle errors from training subprocesses
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)
    
    # Run each training subprocess
    ps = [] 
    # import ctypes
    # best_val_acc = mp.Value('d',0.0)
    # best_val_epoch = mp.Value('i',0)
    # fun_args = tuple(fun_args) + (None,None, )
    
    for i in range(num_proc):
        p_i = mp.Process(
            target=run_swa,
            args=(i, num_proc, error_queue, fun, fun_args, fun_kwargs)
        )
        ps.append(p_i)
        p_i.start()
        error_handler.add_child(p_i.pid)

    # Wait for each subprocess to finish
    for p in ps:
        p.join()

    return
    # return best_val_acc.value, best_val_epoch.value

def multi_proc_run(num_proc, fun, fun_args=(), fun_kwargs={},args=None):
    """Runs a function in a multi-proc setting."""

    # Handle errors from training subprocesses
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)
    
    # Run each training subprocess
    ps = [] 
    import ctypes
    best_val_acc = mp.Value('d',0.0)
    best_val_epoch = mp.Value('i',0)
    fun_args = (best_val_acc,best_val_epoch, ) + fun_args
    for i in range(num_proc):
        p_i = mp.Process(
            target=run,
            args=(i, num_proc, error_queue, fun, fun_args, fun_kwargs)
        )
        ps.append(p_i)
        p_i.start()
        error_handler.add_child(p_i.pid)

    # Wait for each subprocess to finish
    for p in ps:
        p.join()

    return best_val_acc.value, best_val_epoch.value


def multi_proc_run_test(num_proc, fun, fun_args=(), fun_kwargs={},args=None):
    """Runs a function in a multi-proc setting."""

    # Handle errors from training subprocesses
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)
    
    # Run each training subprocess
    ps = [] 
    import ctypes
    test_acc = mp.Value('d',0.0)
    
    fun_args = (test_acc, ) + fun_args
    for i in range(num_proc):
        p_i = mp.Process(
            target=run,
            args=(i, num_proc, error_queue, fun, fun_args, fun_kwargs)
        )
        ps.append(p_i)
        p_i.start()
        error_handler.add_child(p_i.pid)

    # Wait for each subprocess to finish
    for p in ps:
        p.join()

    return test_acc.value
