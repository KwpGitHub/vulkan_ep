#ifndef SELU_H
#define SELU_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Selu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the scaled exponential linear unit function,
`y = gamma * (alpha * e^x - alpha) for x <= 0`, `y = gamma * x for x > 0`,
is applied to the tensor elementwise.

input: Input tensor
output: Output tensor
//*/
//Selu
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha, gamma
//OPTIONAL_PARAMETERS_TYPE: float, float

namespace py = pybind11;

//class stuff
namespace backend {   

    class Selu : public Layer {
        typedef struct {
            float alpha; float gamma;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        float alpha; float gamma;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Selu(std::string n, float alpha, float gamma);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output); 

        ~Selu() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Selu::Selu(std::string n, float alpha, float gamma) : Layer(n) { }
       
    vuh::Device* Selu::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Selu::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.alpha = alpha;
  		binding.gamma = gamma;
 
    }
    
    void Selu::call(std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/selu.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Selu, Layer>(m, "Selu")
            .def(py::init<std::string, float, float> ())
            .def("forward", &Selu::forward)
            .def("init", &Selu::init)
            .def("call", (void (Selu::*) (std::string, std::string)) &Selu::call);
    }
}

#endif

/* PYTHON STUFF

*/

