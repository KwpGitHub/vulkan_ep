#ifndef ARRAYFEATUREEXTRACTOR_H
#define ARRAYFEATUREEXTRACTOR_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Select elements of the input tensor based on the indices passed.<br>
    The indices are applied to the last axes of the tensor.

input: Data to be selected
input: The indices, based on 0 as the first index of any dimension.
output: Selected output data as an array
//*/
//ArrayFeatureExtractor
//INPUTS:                   X_input, Y_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 

namespace py = pybind11;

//class stuff
namespace backend {   

    class ArrayFeatureExtractor : public Layer {
        typedef struct {
            
			
            Shape_t X_input; Shape_t Y_input;
            
            Shape_t Z_output;
            
        } binding_descriptor;

        
        std::string X_input; std::string Y_input;
        
        std::string Z_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ArrayFeatureExtractor(std::string n);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string X_input, std::string Y_input, std::string Z_output); 

        ~ArrayFeatureExtractor() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    ArrayFeatureExtractor::ArrayFeatureExtractor(std::string n) : Layer(n) { }
       
    vuh::Device* ArrayFeatureExtractor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ArrayFeatureExtractor::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
  		binding.Y_input = tensor_dict[Y_input]->shape();
 
		binding.Z_output = tensor_dict[Z_output]->shape();
 

    }
    
    void ArrayFeatureExtractor::call(std::string X_input, std::string Y_input, std::string Z_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/arrayfeatureextractor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[X_input]->data(), *tensor_dict[Y_input]->data(), *tensor_dict[Z_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<ArrayFeatureExtractor, Layer>(m, "ArrayFeatureExtractor")
            .def(py::init<std::string> ())
            .def("forward", &ArrayFeatureExtractor::forward)
            .def("init", &ArrayFeatureExtractor::init)
            .def("call", (void (ArrayFeatureExtractor::*) (std::string, std::string, std::string)) &ArrayFeatureExtractor::call);
    }
}

#endif

/* PYTHON STUFF

*/

