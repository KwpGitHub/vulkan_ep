#ifndef ONEHOTENCODER_H
#define ONEHOTENCODER_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Replace each input element with an array of ones and zeros, where a single
    one is placed at the index of the category that was passed in. The total category count 
    will determine the size of the extra dimension of the output array Y.<br>
    For example, if we pass a tensor with a single value of 4, and a category count of 8, 
    the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.<br>
    This operator assumes every input feature is from the same set of categories.<br>
    If the input is a tensor of float, int32, or double, the data will be cast
    to integers and the cats_int64s category list will be used for the lookups.

input: Data to be encoded.
output: Encoded output data, having one more dimension than X.

*/
//OneHotEncoder
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      cats_int64s, cats_strings, zeros
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*, int

namespace py = pybind11;

//class stuff
namespace backend {   

    class OneHotEncoder : public Layer {
        typedef struct {    
            Shape_t cats_int64s; Tensor* cats_strings; int zeros;
        } parameter_descriptor;  

        typedef struct {
            Tensor* X_input;
            
        } input_desriptor;

        typedef struct {
            Tensor* Y_output;
            
        } output_descriptor;

        typedef struct {
            Shape_t cats_int64s; int zeros;
		Shape_t cats_strings;
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
        OneHotEncoder(std::string, parameter_descriptor _parameter_descriptor);
    
        void forward() { program->run(); }
        
        void call(); 
        void init(); 

        ~OneHotEncoder() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    OneHotEncoder::OneHotEncoder(std::string n, parameter_descriptor _parameter_descriptor) : Layer(n) {
        parameters = _parameter_descriptor;
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/onehotencoder.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
      
    }  

    vuh::Device* OneHotEncoder::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void OneHotEncoder::init() {
		binding.X_input = input.X_input->shape();
 
		binding.Y_output = output.Y_output->shape();
 
		binding.cats_int64s = parameters.cats_int64s;
  		binding.zeros = parameters.zeros;
  		binding.cats_strings = parameters.cats_strings->shape();
 
        program->bind(binding, *parameters.cats_strings->data(), *input.X_input->data(), *output.Y_output->data());
    }
    
    void OneHotEncoder::call(){
       
    }


}



//python stuff
/*namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<OneHotEncoder, Layer>(m, "OneHotEncoder")
            .def("forward", &OneHotEncoder::forward);    
    }
}*/

#endif
