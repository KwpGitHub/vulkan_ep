#ifndef RESIZE_H
#define RESIZE_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

Resize the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).

input: N-D tensor
input: The scale array along each dimension. It takes value greater than 0. If it's less than 1, it's sampling down, otherwise, it's upsampling. The number of elements of 'scales' should be the same as the rank of input 'X'.
output: N-D tensor after resizing
*/

//Resize
//INPUTS:                   X_i, scales_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
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
			
            Shape_t X_i; Shape_t scales_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        int mode;
        std::string X_i; std::string scales_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Resize(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( int _mode); 
        void bind(std::string _X_i, std::string _scales_i, std::string _Y_o); 

        ~Resize() {}
    };

}

#endif

