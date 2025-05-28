#include <sys/prctl.h>
#include <signal.h>

void set_client_pdeathsig() {
    prctl(PR_SET_PDEATHSIG, SIGKILL);
}

#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *set_client_pdeathsig_py(PyObject *self, PyObject *arg) {
    set_client_pdeathsig();
    Py_RETURN_NONE;
}

static PyMethodDef prctl_methods[] = {
    {"set_client_pdeathsig", set_client_pdeathsig_py, METH_NOARGS, NULL},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef prctl_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "prctl",
    .m_size = -1,
    .m_methods = prctl_methods,
};

PyMODINIT_FUNC PyInit_prctl(void) {
    return PyModule_Create(&prctl_module);
}
