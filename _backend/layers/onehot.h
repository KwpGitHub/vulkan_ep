#ifndef ONEHOT_H
#define ONEHOT_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

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
*/

//OneHot
//INPUTS:                   indices_input, depth_input, values_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int

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
        OneHot();
    
        void forward() { program->run(); }
        
        void init( int _axis); 
        void bind(std::string _indices_input, std::string _depth_input, std::string _values_input, std::string _output_output); 

        ~OneHot() {}
    };

    
    void init_layer_OneHot(py::module& m) {
        // py::class_(m, "OneHot");
    }
    

}


#endif

