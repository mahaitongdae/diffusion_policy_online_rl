import functools
import inspect
import os
import dill as pickle
import re
import types
from typing import Callable, List
import numpy as np
import jax, jax.core, jaxlib.xla_client

PATTERN = re.compile(r"^jax")

BLACKLIST = ["pure_callback", "io_callback", "debug_callback"]

def initialize_primitives(pattern: re.Pattern = PATTERN):
    import sys

    registry = {}
    for mod_name, mod in sys.modules.items():
        if pattern.match(mod_name):
            for name, obj in mod.__dict__.items():
                if isinstance(obj, jax.core.Primitive):
                    registry[obj.name] = obj
    return registry

def make_persist(f: Callable):
    try:
        sig = inspect.signature(f)
    except (ValueError, TypeError):
        sig = None

    @functools.wraps(f)
    def make_persist_f(*args) -> PersistFunction:
        in_flat, in_tree = jax.tree.flatten(args)
        in_descr_flat = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in in_flat]
        closed_jaxpr, out_descr = jax.make_jaxpr(f, return_shape=True)(*args)
        out_descr_flat, out_tree = jax.tree_util.tree_flatten(out_descr)

        assert all(x.named_shape == {} and x.sharding is None for x in in_descr_flat)
        assert all(x.named_shape == {} and x.sharding is None for x in out_descr_flat)

        if sig is not None:
            # def f(a, /, b, *args, c, **kwargs): ...
            params = list(sig.parameters.items())
            arg_names = []
            while len(arg_names) < len(args) and len(params) > 0:
                name, param = params[0]
                if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    arg_names.append(name)
                    params.pop(0)
                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                    arg_names.extend(f"*args[{i}]" for i in range(len(args) - len(arg_names)))
                    break
                else:
                    break
            if len(arg_names) != len(args):
                print(f"Warning: Unexpected signature {sig} for {f}, got {len(args)} args")
                arg_names = [f"*args[{i}]" for i in range(len(args))]
        else:
            arg_names = [f"*args[{i}]" for i in range(len(args))]

        return PersistFunction(closed_jaxpr, in_tree, in_descr_flat, out_tree, out_descr_flat, arg_names)

    return make_persist_f

class PersistFunction:
    def __init__(
        self,
        closed_jaxpr: jax.core.ClosedJaxpr,
        in_tree: jax.tree_util.PyTreeDef,
        in_descr_flat: List[jax.ShapeDtypeStruct],
        out_tree: jax.tree_util.PyTreeDef,
        out_descr_flat: List[jax.ShapeDtypeStruct],
        arg_names: List[str],
    ):
        self.closed_jaxpr = closed_jaxpr
        self.in_tree = in_tree
        self.in_descr_flat = in_descr_flat
        self.out_tree = out_tree
        self.out_descr_flat = out_descr_flat
        self.arg_names = arg_names

        jaxpr: jax.core.Jaxpr = closed_jaxpr.jaxpr
        assert not jaxpr.effects
        # Consider add
        # assert jaxpr.debug_info is None, breakpoint()

    def __call__(self, *args):
        in_flat, in_tree = jax.tree_util.tree_flatten(args)
        if in_tree != self.in_tree:
            raise ValueError(f"Expected input tree structure {self.in_tree}, got {in_tree}")
        for arg, descr in zip(in_flat, self.in_descr_flat):
            assert_compatible(arg, descr)
        out_flat = jax.core.eval_jaxpr(self.closed_jaxpr.jaxpr, self.closed_jaxpr.consts, *in_flat)
        for arg, descr in zip(out_flat, self.out_descr_flat):
            assert_compatible(arg, descr)
        return jax.tree_util.tree_unflatten(self.out_tree, out_flat)

    @property
    def in_descr(self):
        return jax.tree_util.tree_unflatten(self.in_tree, self.in_descr_flat)

    @property
    def out_descr(self):
        return jax.tree_util.tree_unflatten(self.out_tree, self.out_descr_flat)

    def save(self, path: os.PathLike):
        pickler = PersistFunctionPickler(open(path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        pickler.dump(self)

    def save_info(self, path: os.PathLike):
        with open(path, "w") as f:
            f.write(self.pretty_print(use_color=False))

    @staticmethod
    def load(path: os.PathLike):
        unpickler = PersistFunctionUnpickler(open(path, "rb"))
        return unpickler.load()

    def pretty_print(self, use_color: bool = True):
        from pprint import PrettyPrinter

        class CustomPrettyPrinter(PrettyPrinter):
            def __init__(self):
                super().__init__(1, 120, None, None, compact=False, sort_dicts=False, underscore_numbers=False)

            def _repr(self, object, *args, **kwargs):
                if isinstance(object, jax.ShapeDtypeStruct):
                    return f"{object.dtype}[{','.join(str(x) for x in object.shape)}]"
                return super()._repr(object, *args, **kwargs)

        def arg_format(name, descr):
            info = pp.pformat(descr).replace("\n", "\n" + " " * (len(name) + 2))
            return f"{name}: {info}"

        assert len(self.arg_names) == len(self.in_descr)
        pp = CustomPrettyPrinter()
        in_info = "\n\n".join(arg_format(name, descr) for name, descr in zip(self.arg_names, self.in_descr))
        out_info = pp.pformat(self.out_descr)
        return f"Exported with Jax version {jax.__version__}\n\n" \
               f"--------- In ---------\n{in_info}\n\n" \
               f"--------- Out ---------\n{out_info}\n\n" \
               f"--------- Jaxpr ---------\n{self.closed_jaxpr.pretty_print(use_color=use_color)}"

class PersistFunctionPickler(pickle.Pickler):
    def persistent_id(self, obj):
        if isinstance(obj, jax.core.Primitive):
            if obj.name in BLACKLIST:
                raise pickle.PickleError(f"Cannot pickle {obj.name}")
            return f"P:{obj.name}"
        elif isinstance(obj, jaxlib.xla_client.Traceback):
            return "T"
        elif isinstance(obj, types.FunctionType) and "<locals>" in obj.__qualname__:
            return "C"
        return None

class PersistFunctionUnpickler(pickle.Unpickler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.registry = initialize_primitives()

    def persistent_load(self, pid):
        if not isinstance(pid, str):
            raise pickle.UnpicklingError(f"Invalid persistent id {pid}")
        if pid == "T" or pid == "C":
            return None
        elif pid.startswith("P:"):
            name = pid[2:]
            if name not in self.registry:
                raise pickle.UnpicklingError(f"Unknown primitive {name}")
            return self.registry[name]
        else:
            raise pickle.UnpicklingError(f"Invalid persistent id {pid}")

def assert_compatible(arg, descr: jax.ShapeDtypeStruct):
    if isinstance(arg, (jax.Array, np.ndarray)):
        assert arg.shape == descr.shape, f"Expected shape {descr.shape}, got {arg.shape}"
        assert arg.dtype == descr.dtype, f"Expected dtype {descr.dtype}, got {arg.dtype}"
    else:
        raise NotImplementedError(f"Unsupported type {type(arg)}")
