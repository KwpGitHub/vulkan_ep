#ifndef MATMUL_H
#define MATMUL_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Matrix product that behaves like numpy.matmul: https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html

input: N-dimensional matrix A
input: N-dimensional matrix B
output: Matrix multiply results from A * B

*/
//MatMul
//INPUTS:                   A_input, B_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class MatMul : public Layer {
        typedef struct {    
            
        } parameter_descriptor;  

        typedef struct {
            Tensor* A_input; Tensor* B_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            
		
            Shape_t A_input; Shape_t B_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        MatMul(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~MatMul() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    MatMul::MatMul(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/matmul.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* MatMul::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void MatMul::init() {
		binding.A_input = input.A_input->shape();
  		binding.B_input = input.B_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 

        program->bind(binding, *input.A_input->data(), *input.B_input->data(), *output.Y_output->data());
    }
    
    void MatMul::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<MatMul, Layer>(m, "MatMul")
            .def("forward", &MatMul::forward);    
    }
}*/

#endif
