#ifndef MATMULINTEGER_H
#define MATMULINTEGER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html.
The production MUST never overflow. The accumulation may overflow if and only if in 32 bits.

input: N-dimensional matrix A
input: N-dimensional matrix B
input: Zero point tensor for input 'A'. It's optional and default value is 0. It could be a scalar or a 1-D tensor, which means a per-tensor or per-row quantization. If it's a 1-D tensor, its number of elements should be equal to the number of rows of input 'A'.
input: Scale tensor for input 'B'. It's optional and default value is 0.  It could be a scalar or a 1-D tensor, which means a per-tensor or per-column quantization. If it's a 1-D tensor, its number of elements should be equal to the number of columns of input 'B'.
output: Matrix multiply results from A * B
//*/
//MatMulInteger
//INPUTS:                   A_input, B_input
//OPTIONAL_INPUTS:          a_zero_point_input_opt, b_zero_point_input_opt
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class MatMulInteger : public Layer {
        typedef struct {
            
			
            Shape_t A_input; Shape_t B_input;
            Shape_t a_zero_point_input_opt; Shape_t b_zero_point_input_opt;
            Shape_t Y_output;
            
        } binding_descriptor;

        
        std::string A_input; std::string B_input;
        std::string a_zero_point_input_opt; std::string b_zero_point_input_opt;
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        MatMulInteger(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string A_input, std::string B_input, std::string a_zero_point_input_opt, std::string b_zero_point_input_opt, std::string Y_output); 

        ~MatMulInteger() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    MatMulInteger::MatMulInteger(std::string n) : Layer(n) { }
       
    vuh::Device* MatMulInteger::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void MatMulInteger::init() {      
    
		binding.A_input = tensor_dict[A_input]->shape();
  		binding.B_input = tensor_dict[B_input]->shape();
  		binding.a_zero_point_input_opt = tensor_dict[a_zero_point_input_opt]->shape();
  		binding.b_zero_point_input_opt = tensor_dict[b_zero_point_input_opt]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 

    }
    
    void MatMulInteger::call(std::string A_input, std::string B_input, std::string a_zero_point_input_opt, std::string b_zero_point_input_opt, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/matmulinteger.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[A_input]->data(), *tensor_dict[B_input]->data(), *tensor_dict[a_zero_point_input_opt]->data(), *tensor_dict[b_zero_point_input_opt]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<MatMulInteger, Layer>(m, "MatMulInteger")
            .def(py::init<std::string> ())
            .def("forward", &MatMulInteger::forward)
            .def("init", &MatMulInteger::init)
            .def("call", (void (MatMulInteger::*) (std::string, std::string, std::string, std::string, std::string)) &MatMulInteger::call);
    }
}

#endif

/* PYTHON STUFF

*/

