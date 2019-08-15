#ifndef IMPUTER_H
#define IMPUTER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Replaces inputs that equal one value with another, leaving all other elements alone.<br>
    This operator is typically used to replace missing values in situations where they have a canonical
    representation, such as -1, 0, NaN, or some extreme value.<br>
    One and only one of imputed_value_floats or imputed_value_int64s should be defined -- floats if the input tensor
    holds floats, integers if the input tensor holds integers. The imputed values must all fit within the
    width of the tensor element type. One and only one of the replaced_value_float or replaced_value_int64 should be defined,
    which one depends on whether floats or integers are being processed.<br>
    The imputed_value attribute length can be 1 element, or it can have one element per input feature.<br>In other words, if the input tensor has the shape [*,F], then the length of the attribute array may be 1 or F. If it is 1, then it is broadcast along the last dimension and applied to each feature.

input: Data to be processed.
output: Imputed output data
//*/
//Imputer
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      imputed_value_floats, imputed_value_int64s, replaced_value_float, replaced_value_int64
//OPTIONAL_PARAMETERS_TYPE: Tensor*, Shape_t, float, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class Imputer : public Layer {
        typedef struct {
            Shape_t imputed_value_int64s; float replaced_value_float; int replaced_value_int64;
			Shape_t imputed_value_floats;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        Shape_t imputed_value_int64s; float replaced_value_float; int replaced_value_int64; std::string imputed_value_floats;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Imputer(std::string n, Shape_t imputed_value_int64s, float replaced_value_float, int replaced_value_int64);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string imputed_value_floats, std::string X_input, std::string Y_output); 

        ~Imputer() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Imputer::Imputer(std::string n, Shape_t imputed_value_int64s, float replaced_value_float, int replaced_value_int64) : Layer(n) { }
       
    vuh::Device* Imputer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Imputer::init() {      
    
		binding.X_input = tensor_dict[X_input]->shape();
 
		binding.Y_output = tensor_dict[Y_output]->shape();
 
		binding.imputed_value_int64s = imputed_value_int64s;
  		binding.replaced_value_float = replaced_value_float;
  		binding.replaced_value_int64 = replaced_value_int64;
  		binding.imputed_value_floats = tensor_dict[imputed_value_floats]->shape();
 
    }
    
    void Imputer::call(std::string imputed_value_floats, std::string X_input, std::string Y_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/imputer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[imputed_value_floats]->data(), *tensor_dict[X_input]->data(), *tensor_dict[Y_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Imputer, Layer>(m, "Imputer")
            .def(py::init<std::string, Shape_t, float, int> ())
            .def("forward", &Imputer::forward)
            .def("init", &Imputer::init)
            .def("call", (void (Imputer::*) (std::string, std::string, std::string)) &Imputer::call);
    }
}

#endif

/* PYTHON STUFF

*/

