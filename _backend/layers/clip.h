#include "../layer.h"
#ifndef CLIP_H
#define CLIP_H 
/*

Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.

input: Input tensor whose elements to be clipped
output: Output tensor with clipped input elements
//*/
//Clip
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      max, min
//OPTIONAL_PARAMETERS_TYPE: float, float

//class stuff
namespace backend {   

    class Clip : public Layer {
        typedef struct {
            float max; float min;
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        float max; float min;
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Clip(std::string n);
    
        void forward() { program->run(); }
        
        void init( float _max,  float _min); 
        void bind(std::string _input_input, std::string _output_output); 

        ~Clip() {}

    };
    
}

#endif

