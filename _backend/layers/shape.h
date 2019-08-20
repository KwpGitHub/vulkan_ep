#ifndef SHAPE_H
#define SHAPE_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.

input: An input tensor.
output: Shape of the input tensor
*/

//Shape
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   shape_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Shape : public Layer {
        typedef struct {
            
			
            Shape_t data_i;
            
            Shape_t shape_o;
            
        } binding_descriptor;

        
        std::string data_i;
        
        std::string shape_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Shape(const std::string& name);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _data_i, std::string _shape_o); 

        ~Shape() {}
    };

}

#endif

