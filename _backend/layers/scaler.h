#ifndef SCALER_H
#define SCALER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.

input: Data to be scaled.
output: Scaled output data.

*/
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
            Tensor* offset; Tensor* scale;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            
		Shape_t offset; Shape_t scale;
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
        Scaler(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~Scaler() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    Scaler::Scaler(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/scaler.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* Scaler::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void Scaler::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.offset = parameters.offset->shape();
  		binding.scale = parameters.scale->shape();
 
        program->bind(binding, *parameters.offset->data(), *parameters.scale->data(), *input.X_input->data(), *output.Y_output->data());
    }
    
    void Scaler::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<Scaler, Layer>(m, "Scaler")
            .def("forward", &Scaler::forward);    
    }
}*/

#endif
