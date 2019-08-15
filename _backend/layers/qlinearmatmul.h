#ifndef QLINEARMATMUL_H
#define QLINEARMATMUL_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
It consumes two quantized input tensors, their scales and zero points, scale and zero point of output, and computes the quantized output.
The quantization formula is y = saturate((x / y_scale) + y_zero_point). For (x / y_scale), it is rounding to nearest ties to even.
Refer to https://en.wikipedia.org/wiki/Rounding for details. Scale and zero point must have same shape.
They must be either scalar (per tensor) or 1-D tensor (per row for 'a' and per column for 'b'). If scale and zero point are 1-D tensor,
the number of elements of scale and zero point tensor of input 'a' and output 'y' should be equal to the number of rows of input 'a',
and the number of elements of scale and zero point tensor of input 'b' should be equal to the number of columns of input 'b'.
Production must never overflow, and accumulation may overflow if and only if in 32 bits.

input: N-dimensional quantized matrix a
input: scale of quantized input a
input: zero point of quantized input a
input: N-dimensional quantized matrix b
input: scale of quantized input b
input: zero point of quantized input b
input: scale of quantized output y
input: zero point of quantized output y
output: Quantized matrix multiply results from a * b
//*/
//QLinearMatMul
//INPUTS:                   a_input, a_scale_input, a_zero_point_input, b_input, b_scale_input, b_zero_point_input, y_scale_input, y_zero_point_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class QLinearMatMul : public Layer {
        typedef struct {
            
			
            Shape_t a_input; Shape_t a_scale_input; Shape_t a_zero_point_input; Shape_t b_input; Shape_t b_scale_input; Shape_t b_zero_point_input; Shape_t y_scale_input; Shape_t y_zero_point_input;
            
            Shape_t y_output;
            
        } binding_descriptor;

        
        std::string a_input; std::string a_scale_input; std::string a_zero_point_input; std::string b_input; std::string b_scale_input; std::string b_zero_point_input; std::string y_scale_input; std::string y_zero_point_input;
        
        std::string y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        QLinearMatMul(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string a_input, std::string a_scale_input, std::string a_zero_point_input, std::string b_input, std::string b_scale_input, std::string b_zero_point_input, std::string y_scale_input, std::string y_zero_point_input, std::string y_output); 

        ~QLinearMatMul() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    QLinearMatMul::QLinearMatMul(std::string n) : Layer(n) { }
       
    vuh::Device* QLinearMatMul::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void QLinearMatMul::init() {      
    
		binding.a_input = tensor_dict[a_input]->shape();
  		binding.a_scale_input = tensor_dict[a_scale_input]->shape();
  		binding.a_zero_point_input = tensor_dict[a_zero_point_input]->shape();
  		binding.b_input = tensor_dict[b_input]->shape();
  		binding.b_scale_input = tensor_dict[b_scale_input]->shape();
  		binding.b_zero_point_input = tensor_dict[b_zero_point_input]->shape();
  		binding.y_scale_input = tensor_dict[y_scale_input]->shape();
  		binding.y_zero_point_input = tensor_dict[y_zero_point_input]->shape();
 
		binding.y_output = tensor_dict[y_output]->shape();
 

    }
    
    void QLinearMatMul::call(std::string a_input, std::string a_scale_input, std::string a_zero_point_input, std::string b_input, std::string b_scale_input, std::string b_zero_point_input, std::string y_scale_input, std::string y_zero_point_input, std::string y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/qlinearmatmul.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[a_input]->data(), *tensor_dict[a_scale_input]->data(), *tensor_dict[a_zero_point_input]->data(), *tensor_dict[b_input]->data(), *tensor_dict[b_scale_input]->data(), *tensor_dict[b_zero_point_input]->data(), *tensor_dict[y_scale_input]->data(), *tensor_dict[y_zero_point_input]->data(), *tensor_dict[y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<QLinearMatMul, Layer>(m, "QLinearMatMul")
            .def(py::init<std::string> ())
            .def("forward", &QLinearMatMul::forward)
            .def("init", &QLinearMatMul::init)
            .def("call", (void (QLinearMatMul::*) (std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string, std::string)) &QLinearMatMul::call);
    }
}

#endif

/* PYTHON STUFF

*/

