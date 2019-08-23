#ifndef CLIP_H
#define CLIP_H 

#include "../layer.h"

/*

Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.

input: Input tensor whose elements to be clipped
output: Output tensor with clipped input elements
*/

//Clip
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      max, min
//OPTIONAL_PARAMETERS_TYPE: float, float


//class stuff
namespace layers {   

    class Clip : public backend::Layer {
        typedef struct {          
            backend::Shape_t input_i;
            
            backend::Shape_t output_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        float max; float min;
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Clip(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( float _max,  float _min); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~Clip() {}
    };
   
}
#endif

