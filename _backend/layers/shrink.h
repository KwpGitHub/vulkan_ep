#ifndef SHRINK_H
#define SHRINK_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

Shrink takes one input data (Tensor<numeric>) and produces one Tensor output,
having same datatype and shape with input. It has two attributes, lambd and
bias. The formula of this operator is: If x < -lambd, y = x + bias;
If x > lambd, y = x - bias; Otherwise, y = 0.

input: The input data as Tensor.
output: The output.
//*/
//Shrink
//INPUTS:                   input_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      bias, lambd
//OPTIONAL_PARAMETERS_TYPE: float, float

namespace py = pybind11;

//class stuff
namespace backend {   

    class Shrink : public Layer {
        typedef struct {
            float bias; float lambd;
			
            Shape_t input_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        float bias; float lambd;
        std::string input_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Shrink(std::string n, float bias, float lambd);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string input_input, std::string output_output); 

        ~Shrink() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Shrink::Shrink(std::string n, float bias, float lambd) : Layer(n) { }
       
    vuh::Device* Shrink::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Shrink::init() {      
    
		binding.input_input = tensor_dict[input_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.bias = bias;
  		binding.lambd = lambd;
 
    }
    
    void Shrink::call(std::string input_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/shrink.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[input_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Shrink, Layer>(m, "Shrink")
            .def(py::init<std::string, float, float> ())
            .def("forward", &Shrink::forward)
            .def("init", &Shrink::init)
            .def("call", (void (Shrink::*) (std::string, std::string)) &Shrink::call);
    }
}

#endif

/* PYTHON STUFF

*/

