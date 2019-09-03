#pragma once
#ifndef MATMULINTEGER_H
#define MATMULINTEGER_H 

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
//INPUTS:                   A_i, B_i
//OPTIONAL_INPUTS:          a_zero_point_i, b_zero_point_i
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class MatMulInteger : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string m_A_i; std::string m_B_i;
        std::string m_a_zero_point_i; std::string m_b_zero_point_i;
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        MatMulInteger(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _A_i, std::string _B_i, std::string _a_zero_point_i, std::string _b_zero_point_i, std::string _Y_o); 
        virtual void build();

        ~MatMulInteger() {}
    };
   
}
#endif

