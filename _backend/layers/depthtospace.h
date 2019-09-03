#pragma once
#ifndef DEPTHTOSPACE_H
#define DEPTHTOSPACE_H 

#include "../layer.h"

/*
DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions.

input: Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth, H is the height and W is the width.
output: Output tensor of [N, C/(blocksize * blocksize), H * blocksize, W * blocksize].
*/

//DepthToSpace
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

    class DepthToSpace : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
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
        DepthToSpace(std::string name);
        
        virtual void forward();        
        virtual void init( int _blocksize); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~DepthToSpace() {}
    };
   
}
#endif

