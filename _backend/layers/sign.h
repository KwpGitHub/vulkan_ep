#ifndef SIGN_H
#define SIGN_H 

#include "../layer.h"

/*

Calculate the sign of the given input tensor element-wise.
If input > 0, output 1. if input < 0, output -1. if input == 0, output 0.

input: Input tensor
output: The sign of the input tensor computed element-wise. It has the same shape and type of the input.
*/

//Sign
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Sign : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;
       

    public:
        Sign(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~Sign() {}
    };
   
}
#endif

