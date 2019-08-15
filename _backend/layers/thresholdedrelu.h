#ifndef THRESHOLDEDRELU_H
#define THRESHOLDEDRELU_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

ThresholdedRelu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = x for x > alpha, y = 0 otherwise,
is applied to the tensor elementwise.

input: Input tensor
output: Output tensor
//*/
//ThresholdedRelu
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha
//OPTIONAL_PARAMETERS_TYPE: float

namespace py = pybind11;

//class stuff
namespace backend {   

    class ThresholdedRelu : public Layer {
        typedef struct {
            float alpha;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        float alpha;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ThresholdedRelu(std::string n, float alpha);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output); 

        ~ThresholdedRelu() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    ThresholdedRelu::ThresholdedRelu(std::string n, float alpha) : Layer(n) { }
       
    vuh::Device* ThresholdedRelu::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ThresholdedRelu::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.alpha = alpha;
 
    }
    
    void ThresholdedRelu::call(std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/thresholdedrelu.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<ThresholdedRelu, Layer>(m, "ThresholdedRelu")
            .def(py::init<std::string, float> ())
            .def("forward", &ThresholdedRelu::forward)
            .def("init", &ThresholdedRelu::init)
            .def("call", (void (ThresholdedRelu::*) (std::string, std::string)) &ThresholdedRelu::call);
    }
}

#endif

/* PYTHON STUFF

*/

