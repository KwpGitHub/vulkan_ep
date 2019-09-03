#pragma once
#ifndef TILE_H
#define TILE_H 

#include "../layer.h"

/*
Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]

input: Input tensor of any shape.
input: 1D int64 tensor of the same length as input's dimension number, includes numbers of repeated copies along input's dimensions.
output: Output tensor of the same dimension and type as tensor input. output_dim[i] = input_dim[i] * repeats[i]
*/

//Tile
//INPUTS:                   input_i, repeats_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Tile : public backend::Layer {
        typedef struct {
            uint32_t input_mask;
            uint32_t output_mask;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string m_input_i; std::string m_repeats_i;
        
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        Tile(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _input_i, std::string _repeats_i, std::string _output_o); 
        virtual void build();

        ~Tile() {}
    };
   
}
#endif

