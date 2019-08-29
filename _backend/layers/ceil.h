#ifndef CEIL_H
#define CEIL_H 

#include "../layer.h"

/*

Ceil takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the ceil is, y = ceil(x), is applied to
the tensor elementwise.

input: Input tensor
output: Output tensor
*/

//Ceil
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Ceil : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
       

    public:
        Ceil(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~Ceil() {}
    };
   
}
#endif

