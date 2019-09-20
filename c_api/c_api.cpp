#include <Python.h>
#include <iostream>
#include <memory>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

#include "kernel/kernel.hpp"
/*
 * Implements an example function.
 */
PyDoc_STRVAR(c_api_example_doc, "example()\
\
Example function");

PyObject *c_api_example(PyObject *self, PyObject *args, PyObject *kwargs) {
    /* Shared references that do not need Py_DECREF before returning. */
    PyObject *obj = NULL;
    int number = 0;

    /* Parse positional and keyword arguments */
	/*static char* keywords[] = { "obj", "number", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", keywords, &obj, &number)) {
        return NULL;
    }

    /* Function implementation starts here */
	int size = 10;
	float* in = new float[size];
	float* out = new float[size];
	for (int i = 0; i < size; ++i) {
		in[i] =  11.1f;
		out[i] = 0.0f;
	}
	std::vector<int> shape;
	auto b_in = (char*)in;
	auto b_out = (char*)out;
	std::cout << sizeof(in) << " " << sizeof(b_in) << std::endl;
	shape.push_back(size);

	auto i = kernel::tensor(b_in, shape, kernel::kFormatFp32);
	auto o = kernel::tensor(b_out, shape, kernel::kFormatFp32);

	std::vector<kernel::tensor> t_in;
	std::vector<kernel::tensor> t_out;
	t_in.push_back(i);
	t_out.push_back(o);


	std::shared_ptr<kernel::layer> l (new kernel::layers::Relu());

	auto t1 = Clock::now();
	l->forward(t_in, t_in, t_out);
	auto t2 = Clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds" << std::endl;

	auto x = (float*)t_in[0].toHost();
	auto y = (float*)t_out[0].toHost();

	std::cout << y[0];
		
    if (number < 0) {
        PyErr_SetObject(PyExc_ValueError, obj);
        return NULL;    /* return NULL indicates error */
    }

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
    PyModule_AddIntConstant(module, "year", 2019);

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
    PyModuleDef_HEAD_INIT,
    "c_api",
    c_api_doc,
    0,              /* m_size */
    NULL,           /* m_methods */
    c_api_slots,
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

PyMODINIT_FUNC PyInit_c_api() {
    return PyModuleDef_Init(&c_api_def);
}
