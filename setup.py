import os, sys
from setuptools import Extension, setup, find_packages

prefix = os.environ["CONDA_PREFIX"]
if sys.platform == "win32":
    include_dirs = [os.path.join(prefix, "Library", "include")]
    library_dirs = [os.path.join(prefix, "Library", "lib")]
else:
    include_dirs = [os.path.join(prefix, "include")]
    library_dirs = [os.path.join(prefix, "lib")]

def create_extension(name):
    return Extension(
        f"relax.{name}",
        [f"src/{name}.c"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
    )

extensions = [
    create_extension("futex"),
    create_extension("spinlock"),
    create_extension("prctl"),
]

setup(
    packages=find_packages(),
    ext_modules=extensions,
)
