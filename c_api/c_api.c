#include <Python.h>
#include "kernel/kernel.hpp"

/*
 * Implements an example function.
 */
PyDoc_STRVAR(c_api_example_doc, "example()\
\
Example function");

PyObject *c_api_example(PyObject *self, PyObject *args, PyObject *kwargs) {
    /* Shared references that do not need Py_DECREF before returning. */

    int number = 0;
    /* Parse positional and keyword arguments */
    /*static char* keywords[] = { "obj", "number", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", keywords, &obj, &number)) {
        return NULL;
    }

    /* Function implementation starts here */
	kernel::tensor t = kernel::tensor();

    Py_RETURN_NONE;
}

/*
 * List of functions to add to c_api in exec_c_api().
 */
static PyMethodDef c_api_functions[] = {
    { "example", (PyCFunction)c_api_example, METH_VARARGS | METH_KEYWORDS, c_api_example_doc },
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Initialize c_api. May be called multiple times, so avoid
 * using static state.
 */
int exec_c_api(PyObject *module) {
    PyModule_AddFunctions(module, c_api_functions);
    PyModule_AddStringConstant(module, "__author__", "mramados");
    PyModule_AddStringConstant(module, "__version__", "1.0.0");

    return 0; /* success */
}

/*
 * Documentation for c_api.
 */
PyDoc_STRVAR(c_api_doc, "The c_api module");


static PyModuleDef_Slot c_api_slots[] = {
    { Py_mod_exec, exec_c_api },
    { 0, NULL }
};

static PyModuleDef c_api_def = {
    PyModuleDef_HEAD_INIT, "c_api", c_api_doc, 0, NULL, c_api_slots, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit_c_api() {
    return PyModuleDef_Init(&c_api_def);
}
