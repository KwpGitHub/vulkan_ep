#include "../layer.h"
#ifndef RESIZE_H
#define RESIZE_H 
/*

Resize the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).

input: N-D tensor
input: The scale array along each dimension. It takes value greater than 0. If it's less than 1, it's sampling down, otherwise, it's upsampling. The number of elements of 'scales' should be the same as the rank of input 'X'.
output: N-D tensor after resizing
//*/
//Resize
//INPUTS:                   X_input, scales_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      mode
//OPTIONAL_PARAMETERS_TYPE: int

//class stuff
namespace backend {   

    class Resize : public Layer {
        typedef struct {
            int mode;
			
            Shape_t X_input; Shape_t scales_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        int mode;
        std::string X_input; std::string scales_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Resize(std::string n);
    
        void forward() { program->run(); }
        
        void init( int _mode); 
        void bind(std::string _X_input, std::string _scales_input, std::string _Y_output); 

        ~Resize() {}

    };
    
}

#endif

