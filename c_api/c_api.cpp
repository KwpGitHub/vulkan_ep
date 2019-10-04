#include <Python.h>
#include <iostream>
#include <memory>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

#include "kernel/kernel.hpp"

PyDoc_STRVAR(c_api_doc, " C api testing function to check layer capabilities");

PyObject *c_api_test(PyObject *self, PyObject *args, PyObject *kwargs) {
    PyObject *obj = NULL;
    int number = 0;

    /* Parse positional and keyword arguments */
	/*static char* keywords[] = { "obj", "number", NULL };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "Oi", keywords, &obj, &number)) {
        return NULL;
    }

    /* Function implementation starts here */


	int size = 1e6;
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

	//geometric
	std::shared_ptr<kernel::layer> acos(new kernel::layers::Acos());
	std::shared_ptr<kernel::layer> acosh(new kernel::layers::Acosh());
	std::shared_ptr<kernel::layer> asin (new kernel::layers::Asin());
	std::shared_ptr<kernel::layer> asinh (new kernel::layers::Asinh());
	std::shared_ptr<kernel::layer> atan (new kernel::layers::Atan());
	std::shared_ptr<kernel::layer> atanh (new kernel::layers::Atanh());		
	std::shared_ptr<kernel::layer> cos(new kernel::layers::Cos());
	std::shared_ptr<kernel::layer> cosh(new kernel::layers::Cosh());
	std::shared_ptr<kernel::layer> sin(new kernel::layers::Sin());
	std::shared_ptr<kernel::layer> sinh(new kernel::layers::Sinh());
	std::shared_ptr<kernel::layer> tan(new kernel::layers::Tan());
	std::shared_ptr<kernel::layer> tanh(new kernel::layers::Tanh());
	
	//activation
	std::shared_ptr<kernel::layer> elu (new kernel::layers::Elu());
	std::shared_ptr<kernel::layer> hardsigmoid(new kernel::layers::Hardsigmoid());
	std::shared_ptr<kernel::layer> leakyrelu (new kernel::layers::LeakyReLU());
	std::shared_ptr<kernel::layer> prelu (new kernel::layers::PReLU());
	std::shared_ptr<kernel::layer> relu(new kernel::layers::Relu());
	std::shared_ptr<kernel::layer> selu(new kernel::layers::Selu());
	std::shared_ptr<kernel::layer> sigmoid (new kernel::layers::Sigmoid());
	std::shared_ptr<kernel::layer> softplus(new kernel::layers::Softplus());
	std::shared_ptr<kernel::layer> softsign (new kernel::layers::Softsign());
	
	//math
	std::shared_ptr<kernel::layer> add(new kernel::layers::Add());
	std::shared_ptr<kernel::layer> sub(new kernel::layers::Sub());
	std::shared_ptr<kernel::layer> mul(new kernel::layers::Mul());
	std::shared_ptr<kernel::layer> pow(new kernel::layers::Pow());
	std::shared_ptr<kernel::layer> round(new kernel::layers::Round());
	std::shared_ptr<kernel::layer> exp(new kernel::layers::Exp());
	std::shared_ptr<kernel::layer> sqrt(new kernel::layers::Sqrt());
	std::shared_ptr<kernel::layer> reciprocal(new kernel::layers::Reciprocal());
	
	//logical
	std::shared_ptr<kernel::layer> and (new kernel::layers::And());
	std::shared_ptr<kernel::layer> or (new kernel::layers::Or());
	std::shared_ptr<kernel::layer> equal(new kernel::layers::Equal());
	std::shared_ptr<kernel::layer> greater(new kernel::layers::Greater());
	std::shared_ptr<kernel::layer> less(new kernel::layers::Less());
	std::shared_ptr<kernel::layer> not (new kernel::layers::Not());
	std::shared_ptr<kernel::layer> xor (new kernel::layers::Xor());
	
	//unary
	std::shared_ptr<kernel::layer> abs(new kernel::layers::Abs());
	std::shared_ptr<kernel::layer> ceil(new kernel::layers::Ceil());
	std::shared_ptr<kernel::layer> clip(new kernel::layers::Clip());
	std::shared_ptr<kernel::layer> floor(new kernel::layers::Floor());
	std::shared_ptr<kernel::layer> log(new kernel::layers::Log());
	std::shared_ptr<kernel::layer> max(new kernel::layers::Max());
	std::shared_ptr<kernel::layer> min(new kernel::layers::Min());
	std::shared_ptr<kernel::layer> mod(new kernel::layers::Mod());
	std::shared_ptr<kernel::layer> neg(new kernel::layers::Neg());
	
	//NN op
	/*
		!Conv 1d 2d 3d
		ConvT 1d 2d 3d


		!MaxPooling 1d 2d 3d
		MaxUnPooling 1d 2d 3d
		!AvgPooling 1d 2d 3d
		LPpooling 1d 2d
		AdaptiveMaxPooling 1d 2d 3d
		AdaptiveAvgPooling 1d 2d 3d


		ReflectivePad 1d 2d
		ReplicationPad 1d 2d 3d
		ZeroPad 2d
		ConstantPad 1d 2d 3d 

		BatchNorm 1d 2d 3d
		GroupNorm 
		InstanceNorm 1d 2d 3d
		LayerNorm
		!LRN - localResponseNorm

		LSTM
		GRU
		RNN

		MatMul

		Dropout 1d 2d 3d 


	*/



	auto t1 = Clock::now();
	relu->forward(t_in, t_in, t_out);
	auto t2 = Clock::now();

	std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds" << std::endl;

	auto x = (float*)t_in[0].toHost();
	auto y = (float*)t_out[0].toHost();

	std::cout << y[0] << std::endl;
		
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
    { "run", (PyCFunction)c_api_test, METH_VARARGS, c_api_doc },
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
PyDoc_STRVAR(c_api_module_doc, "The c_api module");


static PyModuleDef_Slot c_api_slots[] = {
    { Py_mod_exec, exec_c_api },
    { 0, NULL }
};

static PyModuleDef c_api_def = {
    PyModuleDef_HEAD_INIT,  "c_api", c_api_module_doc,
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
