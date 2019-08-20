#ifndef DEPTHTOSPACE_H
#define DEPTHTOSPACE_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*
DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions.

input: Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width.
output: Output tensor of [N, C/(blocksize * blocksize), H * blocksize, W * blocksize].
*/

//DepthToSpace
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               blocksize
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class DepthToSpace : public Layer {
        typedef struct {
            int blocksize;
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int blocksize;
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        DepthToSpace(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( int _blocksize); 
        void bind(std::string _input_input, std::string _output_output); 

        ~DepthToSpace() {}
    };

}

#endif

