import os
import torch
import torch.nn as nn
import numpy as np
import random
import logging
import functools
import sys
from typing import Callable, Any

def get_output(module, input, output):
    return output

def get_input(module, input, output):
    return output

def get_input_output(module, input, output):
    return input, output

def get_module_input_output(module, input, output):
    return module, input, output

def register_hook_fn_to_module(model: nn.Module, module_name: str, hook_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], Any]):
    results_dict = {}
    for name, m in model.named_modules():
        if module_name == name:
            handle = m.register_forward_hook(_hook_fn_cntr(name, results_dict, hook_fn))
            return handle, results_dict

def register_bkw_hook_fn_to_module(model: nn.Module, module_name: str, hook_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], Any]):
    results_dict = {}
    for name, m in model.named_modules():
        if module_name == name:
            handle = m.register_full_backward_hook(_hook_fn_cntr(name, results_dict, hook_fn))
            return handle, results_dict

def register_hook_fn_to_all_modules(model: nn.Module, hook_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], Any]):
    results_dict = {}
    for name, m in model.named_modules():
        _ = m.register_forward_hook(_hook_fn_cntr(name, results_dict, hook_fn))
    return results_dict

def register_bkw_hook_fn_to_all_modules(model: nn.Module, hook_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], Any]):
    results_dict = {}
    for name, m in model.named_modules():
        _ = m.register_full_backward_hook(_hook_fn_cntr(name, results_dict, hook_fn))
    return results_dict

def _hook_fn_cntr(name, activation_dict, hook_fn):
    def hook(model, input, output):
        activation_dict[name] = hook_fn(model, input, output)
    return hook