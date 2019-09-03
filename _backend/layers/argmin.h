#pragma once
#ifndef ARGMIN_H
#define ARGMIN_H 

#include "../layer.h"

/*

Computes the indices of the min elements of the input tensor's element along the 
provided axis. The resulted tensor has the same rank as the input if keepdims equal 1.
If keepdims equal 0, then the resulted tensor have the reduced dimension pruned. 
The type of the output tensor is integer.
input: An input tensor.
output: Reduced output tensor with integer data type.
*/

//ArgMin
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, keepdims
//OPTIONAL_PARAMETERS_TYPE: int, int


//class stuff
namespace layers {   

    class ArgMin : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        int m_axis; int m_keepdims;
        std::string m_data_i;
        
        std::string m_reduced_o;
        

        binding_descriptor   binding;
       

    public:
        ArgMin(std::string name);
        
        virtual void forward();        
        virtual void init( int _axis,  int _keepdims); 
        virtual void bind(std::string _data_i, std::string _reduced_o); 
        virtual void build();

        ~ArgMin() {}
    };
   
}
#endif

