#ifndef SPACETODEPTH_H
#define SPACETODEPTH_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*
SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.

input: Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width.
output: Output tensor of [N, C * blocksize * blocksize, H/blocksize, W/blocksize].
*/

//SpaceToDepth
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

    class SpaceToDepth : public Layer {
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
        SpaceToDepth();
    
        void forward() { program->run(); }
        
        void init( int _blocksize); 
        void bind(std::string _input_input, std::string _output_output); 

        ~SpaceToDepth() {}
    };

    
    void init_layer_SpaceToDepth(py::module& m) {
        // py::class_(m, "SpaceToDepth");
    }
    

}


#endif

