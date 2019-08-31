#ifndef ATAN_H
#define ATAN_H 

#include "../layer.h"

/*

Calculates the arctangent (inverse of tangent) of the given input tensor, element-wise.

input: Input tensor
output: The arctangent of the input tensor computed element-wise
*/

//Atan
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

    class Atan : public backend::Layer {
        typedef struct {
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string m_input_i;
        
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        Atan(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~Atan() {}
    };
   
}
#endif

