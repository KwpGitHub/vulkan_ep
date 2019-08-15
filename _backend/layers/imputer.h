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

*/
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
            Tensor* imputed_value_floats; Shape_t imputed_value_int64s; float replaced_value_float; int replaced_value_int64;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t imputed_value_int64s; float replaced_value_float; int replaced_value_int64;
		Shape_t imputed_value_floats;
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Imputer(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Imputer() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Imputer::Imputer(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/imputer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Imputer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Imputer::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.imputed_value_int64s = parameters.imputed_value_int64s;
  		binding.replaced_value_float = parameters.replaced_value_float;
  		binding.replaced_value_int64 = parameters.replaced_value_int64;
  		binding.imputed_value_floats = parameters.imputed_value_floats->shape();
 
        program->bind(binding, *parameters.imputed_value_floats->data(), *input.X_input->data(), *output.Y_output->data());
    }
    
    void Imputer::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Imputer, Layer>(m, "Imputer")
            .def("forward", &Imputer::forward);    
    }
}*/

#endif
