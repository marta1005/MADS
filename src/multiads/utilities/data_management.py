# import numpy as np


# def cast_to_ndarray(obj):
#     if isinstance(obj, np.ndarray):
#         out = obj
#     elif isinstance(obj, np.ndarray) and obj.ndim == 0:
#         out = np.array([obj.item()])
#     elif not isinstance(obj, np.ndarray):
#         out = np.array([obj])

#     return out
