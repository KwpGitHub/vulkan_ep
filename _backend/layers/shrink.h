#pragma once
#ifndef SHRINK_H
#define SHRINK_H 

#include "../layer.h"

/*

Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
having same datatype and shape with input. It has two attributes, lambd and
bias. The formula of this operator is: If x < -lambd, y = x + bias;
If x > lambd, y = x - bias; Otherwise, y = 0.

input: The input data as Tensor.
output: The output.
*/

//Shrink
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      bias, lambd
//OPTIONAL_PARAMETERS_TYPE: float, float


//class stuff
namespace layers {   

    class Shrink : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        float m_bias; float m_lambd;
        std::string m_input_i;
        
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        Shrink(std::string name);
        
        virtual void forward();        
        virtual void init( float _bias,  float _lambd); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~Shrink() {}
    };
   
}
#endif

