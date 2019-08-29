#ifndef SINH_H
#define SINH_H 

#include "../layer.h"

/*

Calculates the hyperbolic sine of the given input tensor element-wise.

input: Input tensor
output: The hyperbolic sine values of the input tensor computed element-wise
*/

//Sinh
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

    class Sinh : public backend::Layer {
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
        Sinh(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~Sinh() {}
    };
   
}
#endif

