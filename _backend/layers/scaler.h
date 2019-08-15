#ifndef SCALER_H
#define SCALER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.

input: Data to be scaled.
output: Scaled output data.
//*/
//Scaler
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      offset, scale
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Tensor*

namespace py = pybind11;

//class stuff
namespace backend {   

    class Scaler : public Layer {
        typedef struct {
            
			Shape_t offset; Shape_t scale;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        std::string offset; std::string scale;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Scaler(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string offset, std::string scale, std::string X_input, std::string Y_output); 

        ~Scaler() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Scaler::Scaler(std::string n) : Layer(n) { }
       
    vuh::Device* Scaler::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Scaler::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.offset = tensor_dict[offset]->shape();
  		binding.scale = tensor_dict[scale]->shape();
 
    }
    
    void Scaler::call(std::string offset, std::string scale, std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/scaler.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[offset]->data(), *tensor_dict[scale]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Scaler, Layer>(m, "Scaler")
            .def(py::init<std::string> ())
            .def("forward", &Scaler::forward)
            .def("init", &Scaler::init)
            .def("call", (void (Scaler::*) (std::string, std::string, std::string, std::string)) &Scaler::call);
    }
}

#endif

/* PYTHON STUFF

*/

