#pragma once
#ifndef TOPK_H
#define TOPK_H 

#include "../layer.h"

/*

Retrieve the top-K elements along a specified axis. Given an input tensor of
shape [a_1, a_2, ..., a_n, r] and integer argument k, return two outputs:
  -Value tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n]
    which contains the values of the top k elements along the specified axis
  -Index tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] which
   contains the indices of the top k elements (original indices from the input
   tensor).
   
Given two equivalent values, this operator uses the indices along the axis  as
 a tiebreaker. That is, the element with the lower index will appear first.

input: Tensor of shape [a_1, a_2, ..., a_n, r]
input: A 1-D tensor containing a single positive value corresponding to the number of top elements to retrieve
output: Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] containing top K values from the input tensor
output: Tensor of shape [a_1, a_2, ..., a_{axis-1}, k, a_{axis+1}, ... a_n] containing the corresponding input tensor indices for the top K values.
*/

//TopK
//INPUTS:                   X_i, K_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Values_o, Indices_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int


//class stuff
namespace layers {   

    class TopK : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        int m_axis;
        std::string m_X_i; std::string m_K_i;
        
        std::string m_Values_o; std::string m_Indices_o;
        

        binding_descriptor   binding;
       

    public:
        TopK(std::string name);
        
        virtual void forward();        
        virtual void init( int _axis); 
        virtual void bind(std::string _X_i, std::string _K_i, std::string _Values_o, std::string _Indices_o); 
        virtual void build();

        ~TopK() {}
    };
   
}
#endif

