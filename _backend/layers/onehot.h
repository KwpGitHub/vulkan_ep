#ifndef ONEHOT_H
#define ONEHOT_H 
#include <pybind11/pybind11.h>
#include "../layer.h"
/*

    Produces a one-hot tensor based on inputs.
    The locations represented by the index values in the 'indices' input tensor will have 'on_value'
    and the other locations will have 'off_value' in the output tensor, where 'on_value' and 'off_value'
    are specified as part of required input argument 'values', which is a two-element tensor of format
    [off_value, on_value]. The rank of the output tensor will be one greater than the rank of the
    input tensor. The additional dimension is for one-hot representation. The additional dimension will
    be inserted at the position specified by 'axis'. If 'axis' is not specified then then additional
    dimension will be inserted as the innermost dimension, i.e. axis=-1. The size of the additional
    dimension is specified by required scalar input 'depth'. The type of the output tensor is the same
    as the type of the 'values' input. Any entries in the 'indices' input tensor with values outside
    the range [0, depth) will result in one-hot representation with all 'off_value' values in the
    output tensor.

input: Input tensor containing indices. The values must be non-negative integers. Any entries in the 'indices' input tensor with values outside the range [0, depth) will result in one-hot representation with all 'off_value' values in the output tensor.In case 'indices' is of non-integer type, the values will be casted to int64 before use.
input: Scalar specifying the number of classes in one-hot tensor. This is also the size of the one-hot dimension (specified by 'axis' attribute) added on in the output tensor and the values in the 'indices' input tensor are expected to be in the range [0, depth). TheIn case 'depth' is of non-integer type, it will be casted to int64 before use.
input: Rank 1 tensor containing exactly two elements, in the format [off_value, on_value], where 'on_value' is the value used for filling locations specified in 'indices' input tensor, and 'off_value' is the value used for filling locations other than those specified in 'indices' input tensor. 
output: Tensor of rank one greater than input tensor 'indices', i.e. rank(output) = rank(indices) + 1. The data type for the elements of the output tensor is the same as the type of input 'values' is used.
//*/
//OneHot
//INPUTS:                   indices_input, depth_input, values_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int

namespace py = pybind11;

//class stuff
namespace backend {   

    class OneHot : public Layer {
        typedef struct {
            int axis;
			
            Shape_t indices_input; Shape_t depth_input; Shape_t values_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        int axis;
        std::string indices_input; std::string depth_input; std::string values_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        OneHot(std::string n, int axis);
    
        void forward() { program->run(); }
        
        void init(); 
        void call(std::string indices_input, std::string depth_input, std::string values_input, std::string output_output); 

        ~OneHot() {}

    };
    
}


//cpp stuff
namespace backend {    
   
    OneHot::OneHot(std::string n, int axis) : Layer(n) { }
       
    vuh::Device* OneHot::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
    
    void OneHot::init() {      
    
		binding.indices_input = tensor_dict[indices_input]->shape();
  		binding.depth_input = tensor_dict[depth_input]->shape();
  		binding.values_input = tensor_dict[values_input]->shape();
 
		binding.output_output = tensor_dict[output_output]->shape();
 
		binding.axis = axis;
 
    }
    
    void OneHot::call(std::string indices_input, std::string depth_input, std::string values_input, std::string output_output){       
        program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/onehot.spv")).c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind(binding, *tensor_dict[indices_input]->data(), *tensor_dict[depth_input]->data(), *tensor_dict[values_input]->data(), *tensor_dict[output_output]->data());
    }


}



//python stuff
namespace backend {
    PYBIND11_MODULE(_backend, m) {
        py::class_<OneHot, Layer>(m, "OneHot")
            .def(py::init<std::string, int> ())
            .def("forward", &OneHot::forward)
            .def("init", &OneHot::init)
            .def("call", (void (OneHot::*) (std::string, std::string, std::string, std::string)) &OneHot::call);
    }
}

#endif

/* PYTHON STUFF

*/

