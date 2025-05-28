import jax, jax.numpy as jnp
import haiku as hk

def mask_average(x: jax.Array, mask: jax.Array) -> jax.Array:
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1)

def fix_repr(cls):
    """Delete haiku's auto-generated __repr__ method, in favor of dataclass's one"""
    del cls.__repr__
    postinit = getattr(cls, "__post_init__")
    def __post_init__(self):
        postinit(self)
        if hk.running_init():
            print(self)
    cls.__post_init__ = __post_init__
    return cls

def is_broadcastable(src, dst):
    try:
        return jnp.broadcast_shapes(src, dst) == dst
    except ValueError:
        return False

def random_key_from_data(data: jax.Array) -> jax.Array:
    # Create a random key deterministically from data, like hashing
    mean = jnp.mean(data)
    std = jnp.std(data)
    seed = (mean * std).view(jnp.uint32)
    key = jax.random.key(seed)
    return key
