#ifndef HARDSIGMOID_H
#define HARDSIGMOID_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

HardSigmoid takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the HardSigmoid function, y = max(0, min(1, alpha * x + beta)),
is applied to the tensor elementwise.

input: Input tensor
output: Output tensor
//*/
//HardSigmoid
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      alpha, beta
//OPTIONAL_PARAMETERS_TYPE: float, float

namespace py = pybind11;

//class stuff
namespace backend {   

    class HardSigmoid : public Layer {
        typedef struct {
            float alpha; float beta;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        float alpha; float beta;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        HardSigmoid(std::string n, float alpha, float beta);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output); 

        ~HardSigmoid() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    HardSigmoid::HardSigmoid(std::string n, float alpha, float beta) : Layer(n) { }
       
    vuh::Device* HardSigmoid::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void HardSigmoid::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.alpha = alpha;
  		binding.beta = beta;
 
    }
    
    void HardSigmoid::call(std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/hardsigmoid.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<HardSigmoid, Layer>(m, "HardSigmoid")
            .def(py::init<std::string, float, float> ())
            .def("forward", &HardSigmoid::forward)
            .def("init", &HardSigmoid::init)
            .def("call", (void (HardSigmoid::*) (std::string, std::string)) &HardSigmoid::call);
    }
}

#endif

/* PYTHON STUFF

*/

