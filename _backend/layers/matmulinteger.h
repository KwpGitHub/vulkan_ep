#ifndef MATMULINTEGER_H
#define MATMULINTEGER_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

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
//INPUTS:                   A_i, B_i
//OPTIONAL_INPUTS:          a_zero_point_i, b_zero_point_i
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace backend {   

    class MatMulInteger : public Layer {
        typedef struct {
            
			
            Shape_t A_i; Shape_t B_i;
            Shape_t a_zero_point_i; Shape_t b_zero_point_i;
            Shape_t Y_o;
            
        } binding_descriptor;

        
        std::string A_i; std::string B_i;
        std::string a_zero_point_i; std::string b_zero_point_i;
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        MatMulInteger(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init(); 
        virtual void bind(std::string _A_i, std::string _B_i, std::string _a_zero_point_i, std::string _b_zero_point_i, std::string _Y_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/matmulinteger.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[A_i]->data(), *tensor_dict[B_i]->data(), *tensor_dict[a_zero_point_i]->data(), *tensor_dict[b_zero_point_i]->data(), *tensor_dict[Y_o]->data());
        }

        ~MatMulInteger() {}
    };
   
}
#endif

