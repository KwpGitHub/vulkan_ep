#include "../layer.h"
#ifndef SHAPE_H
#define SHAPE_H 
/*

Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.

input: An input tensor.
output: Shape of the input tensor
//*/
//Shape
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   shape_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Shape : public Layer {
        typedef struct {
            
			
            Shape_t data_input;
            
            Shape_t shape_output;
            
        } binding_descriptor;

        
        std::string data_input;
        
        std::string shape_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Shape(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void bind(std::string _data_input, std::string _shape_output); 

        ~Shape() {}

    };
    
}

#endif

