#ifndef CLIP_H
#define CLIP_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Clip operator limits the given input within an interval. The interval is
specified with arguments 'min' and 'max'. They default to
numeric_limits::lowest() and numeric_limits::max() respectively.

input: Input tensor whose elements to be clipped
output: Output tensor with clipped input elements

*/
//Clip
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      max, min
//OPTIONAL_PARAMETERS_TYPE: float, float

namespace py = pybind11;

//class stuff
namespace backend {   

    class Clip : public Layer {
        typedef struct {    
            float max; float min;
        } parameter_descriptor;  

        typedef struct {
            Tensor* input_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* output_output;
            
        } output_descriptor;

        typedef struct {
            float max; float min;
		
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Clip(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Clip() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Clip::Clip(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/clip.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Clip::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Clip::init() {
		binding.input_input = input.input_input->shape();
 
		binding.output_output = output.output_output->shape();
 
		binding.max = parameters.max;
  		binding.min = parameters.min;
 
        program->bind(binding, *input.input_input->data(), *output.output_output->data());
    }
    
    void Clip::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Clip, Layer>(m, "Clip")
            .def("forward", &Clip::forward);    
    }
}*/

#endif
