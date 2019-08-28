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
            uint32_t size; float a;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        float max; float min;
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;
       

    public:
        Clip(std::string name);
        
        virtual void forward();        
        virtual void init( float _max,  float _min); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~Clip() {}
    };
   
}
#endif

