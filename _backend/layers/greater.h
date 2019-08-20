#ifndef GREATER_H
#define GREATER_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Returns the tensor resulted from performing the `greater` logical operation
elementwise on the input tensors `A` and `B` (with Numpy-style broadcasting support).

This operator supports **multidirectional (i.e., Numpy-style) broadcasting**; for more details please check [the doc](Broadcasting.md).

input: First input operand for the logical operator.
input: Second input operand for the logical operator.
output: Result tensor.
*/

//Greater
//INPUTS:                   A_i, B_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   C_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Greater : public Layer {
        typedef struct {
            
			
            Shape_t A_i; Shape_t B_i;
            
            Shape_t C_o;
            
        } binding_descriptor;

        
        std::string A_i; std::string B_i;
        
        std::string C_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Greater(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _A_i, std::string _B_i, std::string _C_o); 

        ~Greater() {}
    };

}

#endif

