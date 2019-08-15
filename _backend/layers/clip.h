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

namespace py = pybind11;

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
        Clip(std::string n, float max, float min);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string output_output); 

        ~Clip() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Clip::Clip(std::string n, float max, float min) : Layer(n) { }
       
    vuh::Device* Clip::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Clip::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.max = max;
  		binding.min = min;
 
    }
    
    void Clip::call(std::string input_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/clip.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Clip, Layer>(m, "Clip")
            .def(py::init<std::string, float, float> ())
            .def("forward", &Clip::forward)
            .def("init", &Clip::init)
            .def("call", (void (Clip::*) (std::string, std::string)) &Clip::call);
    }
}

#endif

/* PYTHON STUFF

*/

