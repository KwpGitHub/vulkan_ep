#include "../layer.h"
#ifndef MATMULINTEGER_H
#define MATMULINTEGER_H 
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

#endif

