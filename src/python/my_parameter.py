import torch.nn.parameter

__new_param__ = torch.nn.parameter.Parameter.__new__

def __new_named_parameter__(*args, **kwargs):
    res = __new_param__(*args, **kwargs)
    res.name = "Whatever"
    print("Created Parameter")
    return res

torch.nn.Parameter.__new__ = __new_named_parameter__
print("Loaded MyParameter")