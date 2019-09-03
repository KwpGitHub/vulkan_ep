#pragma once
#ifndef EXP_H
#define EXP_H 

#include "../layer.h"

/*

Calculates the exponential of the given input tensor, element-wise.

input: Input tensor
output: The exponential of the input tensor computed element-wise
*/

//Exp
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Exp : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string m_input_i;
        
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        Exp(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~Exp() {}
    };
   
}
#endif

