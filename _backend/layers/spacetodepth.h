#ifndef SPACETODEPTH_H
#define SPACETODEPTH_H 

#include "../layer.h"

/*
SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.

input: Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width.
output: Output tensor of [N, C * blocksize * blocksize, H/blocksize, W/blocksize].
*/

//SpaceToDepth
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               blocksize
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class SpaceToDepth : public backend::Layer {
        typedef struct {          
            backend::Shape_t input_i;
            
            backend::Shape_t output_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        int blocksize;
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        SpaceToDepth(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( int _blocksize); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~SpaceToDepth() {}
    };
   
}
#endif

