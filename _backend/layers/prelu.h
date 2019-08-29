#ifndef PRELU_H
#define PRELU_H 

#include "../layer.h"

/*

PRelu takes input data (Tensor<T>) and slope tensor as input, and produces one
output data (Tensor<T>) where the function `f(x) = slope * x for x < 0`,
`f(x) = x for x >= 0`., is applied to the data tensor elementwise.
This operator supports **unidirectional broadcasting** (tensor slope should be unidirectional broadcastable to input tensor X); for more details please check [the doc](Broadcasting.md).
input: Input tensor
input: Slope tensor. The shape of slope can be smaller then first input X; if so, its shape must be unidirectional broadcastable to X
output: Output tensor (same size as X)
*/

//PRelu
//INPUTS:                   X_i, slope_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class PRelu : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string X_i; std::string slope_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
       

    public:
        PRelu(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _X_i, std::string _slope_i, std::string _Y_o); 
        virtual void build();

        ~PRelu() {}
    };
   
}
#endif

