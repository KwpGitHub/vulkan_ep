#include "../layer.h"
#ifndef TILE_H
#define TILE_H 
/*
Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]

input: Input tensor of any shape.
input: 1D int64 tensor of the same length as input's dimension number, includes numbers of repeated copies along input's dimensions.
output: Output tensor of the same dimension and type as tensor input. output_dim[i] = input_dim[i] * repeats[i]
//*/
//Tile
//INPUTS:                   input_input, repeats_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

//class stuff
namespace backend {   

    class Tile : public Layer {
        typedef struct {
            
			
            Shape_t input_input; Shape_t repeats_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        
        std::string input_input; std::string repeats_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Tile(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string repeats_input, std::string output_output); 

        ~Tile() {}

    };
    
}

#endif

