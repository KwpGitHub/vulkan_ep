#pragma once
#ifndef REDUCEL2_H
#define REDUCEL2_H 

#include "../layer.h"

/*

Computes the L2 norm of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
input: An input tensor.
output: Reduced output tensor.
*/

//ReduceL2
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes, keepdims
//OPTIONAL_PARAMETERS_TYPE: std::vector<int>, int


//class stuff
namespace layers {   

    class ReduceL2 : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<int> m_axes; int m_keepdims;
        std::string m_data_i;
        
        std::string m_reduced_o;
        

        binding_descriptor   binding;
       

    public:
        ReduceL2(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<int> _axes,  int _keepdims); 
        virtual void bind(std::string _data_i, std::string _reduced_o); 
        virtual void build();

        ~ReduceL2() {}
    };
   
}
#endif

