#ifndef SQUEEZE_H
#define SQUEEZE_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.

input: Tensors with at least max(dims) dimensions.
output: Reshaped tensor with same data as input.
*/

//Squeeze
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   squeezed_o
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
			
            Shape_t data_i;
            
            Shape_t squeezed_o;
            
        } binding_descriptor;

        Shape_t axes;
        std::string data_i;
        
        std::string squeezed_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Squeeze(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( Shape_t _axes); 
        void bind(std::string _data_i, std::string _squeezed_o); 

        ~Squeeze() {}
    };

}

#endif

