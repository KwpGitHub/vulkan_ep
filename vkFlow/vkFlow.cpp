#include <Python.h>
#include <vector>
#include "documentaion.h"
#include "pipeline/pipeline.h"

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

	auto y = std::vector<float>(128, 1.0f);
	auto x = std::vector<float>(128, 2.0f);


	pipeline::Instance instance = pipeline::Instance();
	pipeline::Device device = instance.devices().at(0);

	pipeline::Array<float> device_x = pipeline::Array<float>(device, y);
	pipeline::Array<float> device_y = pipeline::Array<float>(device, x);

	using Spec = pipeline::typelist<uint32_t>;
	struct Params {uint32_t size, float a};

	auto program = pipeline::Program<Specs, Params>(device, "shader");
	program.grid(128 / 64).spec(64)({ 128, 0.1 }, d_y, d_x);

	d_y.toHost(begin(y));


	Py_RETURN_NONE;
}

/*
 * List of functions to add to vkFlow in exec_vkFlow().
 */
static PyMethodDef vkFlow_functions[] = {
    { "Run", (PyCFunction)Run, METH_NOARGS, vkFlow_Run_doc },
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
