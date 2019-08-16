#include "../layer.h"
#ifndef SQUEEZE_H
#define SQUEEZE_H 
/*

Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.

input: Tensors with at least max(dims) dimensions.
output: Reshaped tensor with same data as input.
//*/
//Squeeze
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   squeezed_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes
//OPTIONAL_PARAMETERS_TYPE: Shape_t

//class stuff
namespace backend {   

    class Squeeze : public Layer {
        typedef struct {
            Shape_t axes;
			
            Shape_t data_input;
            
            Shape_t squeezed_output;
            
        } binding_descriptor;

        Shape_t axes;
        std::string data_input;
        
        std::string squeezed_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Squeeze(std::string n, Shape_t axes);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string data_input, std::string squeezed_output); 

        ~Squeeze() {}

    };
    
}

#endif

