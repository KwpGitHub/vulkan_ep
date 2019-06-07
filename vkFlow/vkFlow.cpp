#include <Python.h>
#include "documentaion.h"
#include "vuh/vuh.h"

/*
 * Implements an example function.
 */

PyObject* set_device(PyObject* self, PyObject* args, PyObject* kwargs) {
	PyObject* obj = NULL;
	int number = 0;

	static char* keywords[] = {"deviceNum", NULL };
	if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", keywords, &obj, &number)) {
		return NULL;
	}
	if (number < 0) {
		PyErr_SetObject(PyExc_ValueError, obj);
		return NULL;    /* return NULL indicates error */
	}
	int deviceProc = PY_UINT32_T(obj);
	Py_RETURN_NONE;
}

PyObject* Run(PyObject* self, PyObject* args, PyObject* kwargs) {
	
	PyObject* obj = NULL;
	int number = 0;


	Py_RETURN_NONE;
}

/*
 * List of functions to add to vkFlow in exec_vkFlow().
 */
static PyMethodDef vkFlow_functions[] = {
    { "Run", (PyCFunction)Run, METH_VARARGS | METH_KEYWORDS, vkFlow_example_doc },
	{ "set_device", (PyCFunction)set_device, METH_VARARGS | METH_KEYWORDS, vkFlow_example_doc },
    { NULL, NULL, 0, NULL } /* marks end of array */
};

/*
 * Initialize vkFlow. May be called multiple times, so avoid
 * using static state.
 */
int exec_vkFlow(PyObject *module) {
    PyModule_AddFunctions(module, vkFlow_functions);
    PyModule_AddStringConstant(module, "__author__", "monish");
    PyModule_AddStringConstant(module, "__version__", "1.0.0");
   


    return 0; /* success */
}

/*
 * Documentation for vkFlow.
 */
PyDoc_STRVAR(vkFlow_doc, "The vkFlow module");


static PyModuleDef_Slot vkFlow_slots[] = {
    { Py_mod_exec, exec_vkFlow }, { 0, NULL }
};

static PyModuleDef vkFlow_def = {
    PyModuleDef_HEAD_INIT,
    "vkFlow",
    vkFlow_doc,
    0,              /* m_size */
    NULL,           /* m_methods */
    vkFlow_slots,
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit_vkFlow() {
	
    return PyModuleDef_Init(&vkFlow_def);
}
