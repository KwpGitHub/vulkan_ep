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
            
            
            backend::Shape_t output_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::vector<float> value;
        
        
        std::string output_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Constant(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( std::vector<float> _value); 
        virtual void bind(std::string _output_o); 
        virtual void build();

        ~Constant() {}
    };
   
}
#endif

