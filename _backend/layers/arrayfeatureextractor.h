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

*/
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
            
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input; Tensor* Y_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Z_output;
            
        } output_descriptor;

        typedef struct {
            
		
            Shape_t X_input; Shape_t Y_input;
            
            Shape_t Z_output;
            
        } binding_descriptor;

        parameter_descriptor parameters;
        input_desriptor      input;
        output_descriptor    output;
        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ArrayFeatureExtractor(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~ArrayFeatureExtractor() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    ArrayFeatureExtractor::ArrayFeatureExtractor(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/arrayfeatureextractor.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* ArrayFeatureExtractor::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void ArrayFeatureExtractor::init() {
		binding.X_input = input.X_input->shape();
  		binding.Y_input = input.Y_input->shape();
 
		binding.Z_output = output.Z_output->shape();
 

        program->bind(binding, *input.X_input->data(), *input.Y_input->data(), *output.Z_output->data());
    }
    
    void ArrayFeatureExtractor::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<ArrayFeatureExtractor, Layer>(m, "ArrayFeatureExtractor")
            .def("forward", &ArrayFeatureExtractor::forward);    
    }
}*/

#endif
