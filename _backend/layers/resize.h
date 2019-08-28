#ifndef RESIZE_H
#define RESIZE_H 

#include "../layer.h"

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
//OPTIONAL_PARAMETERS_TYPE: std::string


//class stuff
namespace layers {   

    class Resize : public backend::Layer {
        typedef struct {
            uint32_t size; float a;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::string mode;
        std::string X_i; std::string scales_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
       

    public:
        Resize(std::string name);
        
        virtual void forward();        
        virtual void init( std::string _mode); 
        virtual void bind(std::string _X_i, std::string _scales_i, std::string _Y_o); 
        virtual void build();

        ~Resize() {}
    };
   
}
#endif

