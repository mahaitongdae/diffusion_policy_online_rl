#define _DEFAULT_SOURCE

#include <linux/futex.h>
#include <sys/syscall.h>
#include <sched.h>
#include <stdatomic.h>
#include <unistd.h>
#include <stdint.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

struct lock_t
{
    atomic_uint futex;
    atomic_uint count;
};
#define LOCK_FUTEX(lock) (&(lock->futex))
#define LOCK_COUNT(lock) (&(lock->count))

int futex_server_wait(struct lock_t *lock, int target) {
    atomic_uint *count = LOCK_COUNT(lock);
    size_t i = 0;
    while (atomic_load(count) < target) {
        // call PyErr_CheckSignals to handle KeyboardInterrupt
        if (i++ % 100 == 0) {
            if (PyErr_CheckSignals() != 0) {
                return -1;
            }
        }
        sched_yield();
    }
    atomic_store(count, 0);
    return 0;
}

void futex_server_notify(struct lock_t *lock, int value) {
    atomic_uint *futex = LOCK_FUTEX(lock);
    atomic_store(futex, value);
    syscall(SYS_futex, futex, FUTEX_WAKE, INT32_MAX, NULL, NULL, 0);
}

uint32_t futex_client_wait(struct lock_t *lock, int value) {
    atomic_uint *futex = LOCK_FUTEX(lock);
    syscall(SYS_futex, futex, FUTEX_WAIT, value, NULL, NULL, 0);
    return atomic_load(futex);
}

void futex_client_notify(struct lock_t *lock) {
    atomic_uint *count = LOCK_COUNT(lock);
    atomic_fetch_add(count, 1);
}

static PyObject *futex_server_wait_py(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "futex_server_wait() takes exactly 2 arguments");
        return NULL;
    }
    atomic_uint *count = (atomic_uint *)PyLong_AsVoidPtr(args[0]);
    if (count == NULL) {
        PyErr_SetString(PyExc_TypeError, "futex_server_wait() requires a valid pointer");
        return NULL;
    }
    int target = (int)PyLong_AsLong(args[1]);
    int status = futex_server_wait(count, target);
    if (status == -1) {
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *futex_server_notify_py(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "futex_server_notify() takes exactly 2 arguments");
        return NULL;
    }
    atomic_uint *addr = (atomic_uint *)PyLong_AsVoidPtr(args[0]);
    if (addr == NULL) {
        PyErr_SetString(PyExc_TypeError, "futex_server_notify() requires a valid pointer");
        return NULL;
    }
    int value = (int)PyLong_AsLong(args[1]);
    futex_server_notify(addr, value);
    Py_RETURN_NONE;
}

static PyObject *futex_client_wait_py(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "futex_client_wait() takes exactly 2 arguments");
        return NULL;
    }
    atomic_uint *addr = (atomic_uint *)PyLong_AsVoidPtr(args[0]);
    if (addr == NULL) {
        PyErr_SetString(PyExc_TypeError, "futex_client_wait() requires a valid pointer");
        return NULL;
    }
    int value = (int)PyLong_AsLong(args[1]);
    uint32_t new = futex_client_wait(addr, value);
    return PyLong_FromUnsignedLong(new);
}

static PyObject *futex_client_notify_py(PyObject *self, PyObject *arg) {
    atomic_uint *count = (atomic_uint *)PyLong_AsVoidPtr(arg);
    if (count == NULL) {
        PyErr_SetString(PyExc_TypeError, "futex_client_notify() requires a valid pointer");
        return NULL;
    }
    futex_client_notify(count);
    Py_RETURN_NONE;
}

static PyMethodDef futex_methods[] = {
    {"futex_server_wait", (PyCFunction)futex_server_wait_py, METH_FASTCALL, NULL},
    {"futex_server_notify", (PyCFunction)futex_server_notify_py, METH_FASTCALL, NULL},
    {"futex_client_wait", (PyCFunction)futex_client_wait_py, METH_FASTCALL, NULL},
    {"futex_client_notify", (PyCFunction)futex_client_notify_py, METH_O, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef futex_module = {
    PyModuleDef_HEAD_INIT,
    "futex",
    NULL,
    -1,
    futex_methods
};

PyMODINIT_FUNC PyInit_futex(void) {
    return PyModule_Create(&futex_module);
}
