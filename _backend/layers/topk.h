#ifndef TOPK_H
#define TOPK_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

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
namespace backend {   

    class TopK : public Layer {
        typedef struct {
            int axis;
			
            Shape_t X_i; Shape_t K_i;
            
            Shape_t Values_o; Shape_t Indices_o;
            
        } binding_descriptor;

        int axis;
        std::string X_i; std::string K_i;
        
        std::string Values_o; std::string Indices_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        TopK(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( int _axis); 
        void bind(std::string _X_i, std::string _K_i, std::string _Values_o, std::string _Indices_o); 

        ~TopK() {}
    };

}

#endif

