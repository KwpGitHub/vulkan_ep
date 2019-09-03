#pragma once
#ifndef SUB_H
#define SUB_H 

#include "../layer.h"

/*

Performs element-wise binary subtraction (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: First operand.
input: Second operand.
output: Result, has same element type as two inputs
*/

//Sub
//INPUTS:                   A_i, B_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   C_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Sub : public backend::Layer {
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
        
        std::string m_C_o;
        

        binding_descriptor   binding;
       

    public:
        Sub(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _A_i, std::string _B_i, std::string _C_o); 
        virtual void build();

        ~Sub() {}
    };
   
}
#endif

