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

*/
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
            
        } parameter_descriptor;  

        typedef struct {
            Tensor* A_input; Tensor* B_input;
            Tensor* a_zero_point_input_opt; Tensor* b_zero_point_input_opt;
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            
		
            Shape_t A_input; Shape_t B_input;
            Shape_t a_zero_point_input_opt; Shape_t b_zero_point_input_opt;
            Shape_t Y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        MatMulInteger(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~MatMulInteger() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    MatMulInteger::MatMulInteger(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/matmulinteger.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* MatMulInteger::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void MatMulInteger::init() {
		binding.A_input = input.A_input->shape();
  		binding.B_input = input.B_input->shape();
  		binding.a_zero_point_input_opt = input.a_zero_point_input_opt->shape();
  		binding.b_zero_point_input_opt = input.b_zero_point_input_opt->shape();
 
		binding.Y_output = output.Y_output->shape();
 

        program->bind(binding, *input.A_input->data(), *input.B_input->data(), *input.a_zero_point_input_opt->data(), *input.b_zero_point_input_opt->data(), *output.Y_output->data());
    }
    
    void MatMulInteger::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<MatMulInteger, Layer>(m, "MatMulInteger")
            .def("forward", &MatMulInteger::forward);    
    }
}*/

#endif
