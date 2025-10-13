import sys
from pympler import asizeof
import inspect


def measuresize(fl_fit):
    size_with_sys = sys.getsizeof(fl_fit)
    size_with_pympler = asizeof.asizeof(fl_fit)
    print(f"sys.getsizeof():{size_with_sys} bytes")
    print(f"pympler.asizeof():{size_with_pympler} bytes")

def print_local_variables():
    current_frame = inspect.currentframe()
    caller_frame = inspect.getouterframes(current_frame, 2)
    local_vars = caller_frame[1].frame.f_locals
    for var_name, var_value in local_vars.items():
        if not var_name.startswith("__") and not inspect.isclass(var_value) and not inspect.ismodule(var_value) and not inspect.isfunction(var_value):
            print(f"{var_name}: {var_value}")
            
import h5py
def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print(f"    Attribute: {key} = {val}")
def print_hdf5_keys(filename):
    with h5py.File(filename, 'r') as f:
        try:
            for key, val in f['Header'].attrs.items():
                print(f"Header: {key} = {val}")
        except:
            print('Header not exist')
        f.visititems(print_attrs)