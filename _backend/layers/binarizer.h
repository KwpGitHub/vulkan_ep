#ifndef BINARIZER_H
#define BINARIZER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.

input: Data to be binarized
output: Binarized output data
//*/
//Binarizer
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      threshold
//OPTIONAL_PARAMETERS_TYPE: float

namespace py = pybind11;

//class stuff
namespace backend {   

    class Binarizer : public Layer {
        typedef struct {
            float threshold;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        float threshold;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Binarizer(std::string n, float threshold);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_output); 

        ~Binarizer() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Binarizer::Binarizer(std::string n, float threshold) : Layer(n) { }
       
    vuh::Device* Binarizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Binarizer::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.threshold = threshold;
 
    }
    
    void Binarizer::call(std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/binarizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Binarizer, Layer>(m, "Binarizer")
            .def(py::init<std::string, float> ())
            .def("forward", &Binarizer::forward)
            .def("init", &Binarizer::init)
            .def("call", (void (Binarizer::*) (std::string, std::string)) &Binarizer::call);
    }
}

#endif

/* PYTHON STUFF

*/

