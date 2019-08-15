#ifndef BINARIZER_H
#define BINARIZER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.

input: Data to be binarized
output: Binarized output data

*/
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
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            float threshold;
		
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
        Binarizer(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Binarizer() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Binarizer::Binarizer(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/binarizer.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Binarizer::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Binarizer::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.threshold = parameters.threshold;
 
        program->bind(binding, *input.X_input->data(), *output.Y_output->data());
    }
    
    void Binarizer::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Binarizer, Layer>(m, "Binarizer")
            .def("forward", &Binarizer::forward);    
    }
}*/

#endif
