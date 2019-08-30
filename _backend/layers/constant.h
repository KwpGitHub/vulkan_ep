#ifndef CONSTANT_H
#define CONSTANT_H 

#include "../layer.h"

/*
A constant tensor.

output: Output tensor containing the same value of the provided tensor.
*/

//Constant
//INPUTS:                   
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               value
//PARAMETER_TYPES:          std::vector<float>
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Constant : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<float> m_value;
        
        
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        Constant(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<float> _value); 
        virtual void bind(std::string _output_o); 
        virtual void build();

        ~Constant() {}
    };
   
}
#endif

