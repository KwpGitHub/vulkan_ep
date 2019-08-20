#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Insert single-dimensional entries to the shape of a tensor.
Takes one required argument `axes`, a list of dimensions that will be inserted.
Dimension indices in `axes` are as seen in the output tensor. For example:
  Given a tensor such that tensor with shape [3, 4, 5], then
  Unsqueeze(tensor, axes=[0, 4]) has shape [1, 3, 4, 5, 1]

input: Original tensor
output: Reshaped tensor with same data as input.
*/

//Unsqueeze
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   expanded_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               axes
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Unsqueeze : public Layer {
        typedef struct {
            Shape_t axes;
			
            Shape_t data_input;
            
            Shape_t expanded_output;
            
        } binding_descriptor;

        Shape_t axes;
        std::string data_input;
        
        std::string expanded_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Unsqueeze(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( Shape_t _axes); 
        void bind(std::string _data_input, std::string _expanded_output); 

        ~Unsqueeze() {}
    };

}

#endif

