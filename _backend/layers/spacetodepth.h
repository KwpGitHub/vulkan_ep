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
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        int m_blocksize;
        std::string m_input_i;
        
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        SpaceToDepth(std::string name);
        
        virtual void forward();        
        virtual void init( int _blocksize); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~SpaceToDepth() {}
    };
   
}
#endif

