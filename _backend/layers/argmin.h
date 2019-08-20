#ifndef ARGMIN_H
#define ARGMIN_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

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
namespace backend {   

    class ArgMin : public Layer {
        typedef struct {
            int axis; int keepdims;
			
            Shape_t data_i;
            
            Shape_t reduced_o;
            
        } binding_descriptor;

        int axis; int keepdims;
        std::string data_i;
        
        std::string reduced_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ArgMin(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( int _axis,  int _keepdims); 
        void bind(std::string _data_i, std::string _reduced_o); 

        ~ArgMin() {}
    };

}

#endif

