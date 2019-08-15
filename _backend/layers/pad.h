#ifndef PAD_H
#define PAD_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Given `data` tensor, pads, mode, and value.
Example:
  Insert 0 pads to the beginning of the second dimension.
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  pads = [0, 2, 0, 0]
  output = [
      [
          [0.0, 0.0, 1.0, 1.2],
          [0.0, 0.0, 2.3, 3.4],
          [0.0, 0.0, 4.5, 5.7],
      ],
  ]

input: Input tensor.
output: Tensor after padding.
//*/
//Pad
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               pads
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      mode, value
//OPTIONAL_PARAMETERS_TYPE: int, float

namespace py = pybind11;

//class stuff
namespace backend {   

    class Pad : public Layer {
        typedef struct {
            Shape_t pads; int mode; float value;
			
            Shape_t data_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        Shape_t pads; int mode; float value;
        std::string data_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Pad(std::string n, Shape_t pads, int mode, float value);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string data_input, std::string output_output); 

        ~Pad() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Pad::Pad(std::string n, Shape_t pads, int mode, float value) : Layer(n) { }
       
    vuh::Device* Pad::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Pad::init() {      
    
		binding.data_input = tensor_dict[data_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.pads = pads;
  		binding.mode = mode;
  		binding.value = value;
 
    }
    
    void Pad::call(std::string data_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/pad.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[data_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Pad, Layer>(m, "Pad")
            .def(py::init<std::string, Shape_t, int, float> ())
            .def("forward", &Pad::forward)
            .def("init", &Pad::init)
            .def("call", (void (Pad::*) (std::string, std::string)) &Pad::call);
    }
}

#endif

/* PYTHON STUFF

*/

