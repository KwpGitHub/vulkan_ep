#ifndef CLIP_H
#define CLIP_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

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
namespace backend {   

    class Clip : public Layer {
        typedef struct {
            float max; float min;
			
            Shape_t input_i;
            
            Shape_t output_o;
            
        } binding_descriptor;

        float max; float min;
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Clip(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( float _max,  float _min); 
        virtual void bind(std::string _input_i, std::string _output_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/clip.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[input_i]->data(), *tensor_dict[output_o]->data());
        }

        ~Clip() {}
    };
   
}
#endif

