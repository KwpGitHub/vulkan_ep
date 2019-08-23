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
            backend::Shape_t input_i; backend::Shape_t repeats_i;
            
            backend::Shape_t output_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        
        std::string input_i; std::string repeats_i;
        
        std::string output_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Tile(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init(); 
        virtual void bind(std::string _input_i, std::string _repeats_i, std::string _output_o); 
        virtual void build();

        ~Tile() {}
    };
   
}
#endif

