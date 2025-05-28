#include <sched.h>
#include <stdatomic.h>
#include <unistd.h>

void spinlock_server_wait(atomic_char *guard, int n) {
    for (int i = 0; i < n; i++) {
        while (atomic_load(guard + i) > 0) {
            sched_yield();
        }
    }
}

void spinlock_server_wait_mask(atomic_char *guard, int n, char* mask) {
    for (int i = 0; i < n; i++) {
        if (mask[i] == 0) {
            continue;
        }
        while (atomic_load(guard + i) > 0) {
            sched_yield();
        }
    }
}

void spinlock_server_notify(atomic_char *guard, int n, char value) {
    for (int i = 0; i < n; i++) {
        atomic_store(guard + i, value);
    }
}

void spinlock_server_notify_mask(atomic_char *guard, int n, char* mask, char value) {
    for (int i = 0; i < n; i++) {
        if (mask[i] == 0) {
            continue;
        }
        atomic_store(guard + i, value);
    }
}

void spinlock_client_wait(atomic_char *guard) {
    while (atomic_load(guard) < 0) {
        sched_yield();
        // usleep(0);
    }
}

void spinlock_client_notify(atomic_char *guard, char value) {
    atomic_store(guard, value);
}

#define PY_SSIZE_T_CLEAN
#include <Python.h>

static atomic_char *atomic_from_memoryview(PyObject *obj) {
    if (!PyMemoryView_Check(obj)) {
        return NULL;
    }
    Py_buffer *view = PyMemoryView_GET_BUFFER(obj);
    if (view->len != 1 || view->itemsize != 1) {
        return NULL;
    }
    return (atomic_char *)view->buf;
}

static PyObject *spinlock_server_wait_py(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "spinlock_server_wait() takes exactly 2 arguments");
        return NULL;
    }
    atomic_char *guard = (atomic_char *)PyLong_AsVoidPtr(args[0]);
    if (guard == NULL) {
        PyErr_SetString(PyExc_TypeError, "spinlock_server_wait() requires a valid pointer");
        return NULL;
    }
    int n = (int)PyLong_AsLong(args[1]);
    spinlock_server_wait(guard, n);
    Py_RETURN_NONE;
}

static PyObject *spinlock_server_wait_mask_py(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "spinlock_server_wait_mask() takes exactly 3 arguments");
        return NULL;
    }
    atomic_char *guard = (atomic_char *)PyLong_AsVoidPtr(args[0]);
    if (guard == NULL) {
        PyErr_SetString(PyExc_TypeError, "spinlock_server_wait_mask() requires a valid pointer");
        return NULL;
    }
    int n = (int)PyLong_AsLong(args[1]);
    char *mask = (char *)PyLong_AsVoidPtr(args[2]);
    if (mask == NULL) {
        PyErr_SetString(PyExc_TypeError, "spinlock_server_wait_mask() requires a valid pointer");
        return NULL;
    }
    spinlock_server_wait_mask(guard, n, mask);
    Py_RETURN_NONE;
}

static PyObject *spinlock_server_notify_py(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "spinlock_server_notify() takes exactly 3 arguments");
        return NULL;
    }
    atomic_char *guard = (atomic_char *)PyLong_AsVoidPtr(args[0]);
    if (guard == NULL) {
        PyErr_SetString(PyExc_TypeError, "spinlock_server_notify() requires a valid pointer");
        return NULL;
    }
    int n = (int)PyLong_AsLong(args[1]);
    char value = (char)PyLong_AsLong(args[2]);
    spinlock_server_notify(guard, n, value);
    Py_RETURN_NONE;
}

static PyObject *spinlock_server_notify_mask_py(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 4) {
        PyErr_SetString(PyExc_TypeError, "spinlock_server_notify_mask() takes exactly 4 arguments");
        return NULL;
    }
    atomic_char *guard = (atomic_char *)PyLong_AsVoidPtr(args[0]);
    if (guard == NULL) {
        PyErr_SetString(PyExc_TypeError, "spinlock_server_notify_mask() requires a valid pointer");
        return NULL;
    }
    int n = (int)PyLong_AsLong(args[1]);
    char *mask = (char *)PyLong_AsVoidPtr(args[2]);
    if (mask == NULL) {
        PyErr_SetString(PyExc_TypeError, "spinlock_server_notify_mask() requires a valid pointer");
        return NULL;
    }
    char value = (char)PyLong_AsLong(args[3]);
    spinlock_server_notify_mask(guard, n, mask, value);
    Py_RETURN_NONE;
}

static PyObject *spinlock_client_wait_py(PyObject *self, PyObject *arg) {
    atomic_char *guard = (atomic_char *)PyLong_AsVoidPtr(arg);
    if (guard == NULL) {
        PyErr_SetString(PyExc_TypeError, "spinlock_client_wait() requires a valid pointer");
        return NULL;
    }
    spinlock_client_wait(guard);
    Py_RETURN_NONE;
}

static PyObject *spinlock_client_notify_py(PyObject *self, PyObject *const *args, Py_ssize_t nargs) {
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "spinlock_client_notify() takes exactly 2 arguments");
        return NULL;
    }
    atomic_char *guard = (atomic_char *)PyLong_AsVoidPtr(args[0]);
    if (guard == NULL) {
        PyErr_SetString(PyExc_TypeError, "spinlock_client_notify() requires a valid pointer");
        return NULL;
    }
    char value = (char)PyLong_AsLong(args[1]);
    spinlock_client_notify(guard, value);
    Py_RETURN_NONE;
}

static PyMethodDef spinlock_methods[] = {
    {"spinlock_server_wait", (PyCFunction)spinlock_server_wait_py, METH_FASTCALL, NULL},
    {"spinlock_server_wait_mask", (PyCFunction)spinlock_server_wait_mask_py, METH_FASTCALL, NULL},
    {"spinlock_server_notify", (PyCFunction)spinlock_server_notify_py, METH_FASTCALL, NULL},
    {"spinlock_server_notify_mask", (PyCFunction)spinlock_server_notify_mask_py, METH_FASTCALL, NULL},
    {"spinlock_client_wait", (PyCFunction)spinlock_client_wait_py, METH_O, NULL},
    {"spinlock_client_notify", (PyCFunction)spinlock_client_notify_py, METH_FASTCALL, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef spinlock_module = {
    PyModuleDef_HEAD_INIT,
    "spinlock",
    NULL,
    -1,
    spinlock_methods
};

PyMODINIT_FUNC PyInit_spinlock(void) {
    return PyModule_Create(&spinlock_module);
}
