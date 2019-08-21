#ifndef LABELENCODER_H
#define LABELENCODER_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

    Maps each element in the input tensor to another value.<br>
    The mapping is determined by the two parallel attributes, 'keys_*' and
    'values_*' attribute. The i-th value in the specified 'keys_*' attribute
    would be mapped to the i-th value in the specified 'values_*' attribute. It
    implies that input's element type and the element type of the specified
    'keys_*' should be identical while the output type is identical to the
    specified 'values_*' attribute. If an input element can not be found in the
    specified 'keys_*' attribute, the 'default_*' that matches the specified
    'values_*' attribute may be used as its output value.<br>
    Let's consider an example which maps a string tensor to an integer tensor.
    Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
    and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
    "Sally"] would be mapped to [-1, 5, 5, 6, 6].<br>
    Since this operator is an one-to-one mapping, its input and output shapes
    are the same. Notice that only one of 'keys_*'/'values_*' can be set.<br>
    For key look-up, bit-wise comparison is used so even a float NaN can be
    mapped to a value in 'values_*' attribute.<br>

input: Input data. It can be either tensor or scalar.
output: Output data.
*/

//LabelEncoder
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      default_float, default_int64, default_string, keys_floats, keys_int64s, keys_strings, values_floats, values_int64s, values_strings
//OPTIONAL_PARAMETERS_TYPE: float, int, int, Tensor*, Shape_t, Tensor*, Tensor*, Shape_t, Tensor*


//class stuff
namespace backend {   

    class LabelEncoder : public Layer {
        typedef struct {
            float default_float; int default_int64; int default_string; Shape_t keys_int64s; Shape_t values_int64s;
			Shape_t keys_floats; Shape_t keys_strings; Shape_t values_floats; Shape_t values_strings;
            Shape_t X_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        float default_float; int default_int64; int default_string; Shape_t keys_int64s; Shape_t values_int64s; std::string keys_floats; std::string keys_strings; std::string values_floats; std::string values_strings;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        LabelEncoder(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( float _default_float,  int _default_int64,  int _default_string,  Shape_t _keys_int64s,  Shape_t _values_int64s); 
        virtual void bind(std::string _keys_floats, std::string _keys_strings, std::string _values_floats, std::string _values_strings, std::string _X_i, std::string _Y_o); 

        virtual void build(){
            program = new vuh::Program<Specs, binding_descriptor>(*_get_device(), std::string(file_path + std::string("/shaders/bin/labelencoder.spv")).c_str());
            program->grid(1024 / PROCESSKERNEL_SIZE, 1024 / PROCESSKERNEL_SIZE, 64 / PROCESSKERNEL_SIZE);
            program->spec(64, 64, 64);
            //program->bind(binding, *tensor_dict[keys_floats]->data(), *tensor_dict[keys_strings]->data(), *tensor_dict[values_floats]->data(), *tensor_dict[values_strings]->data(), *tensor_dict[X_i]->data(), *tensor_dict[Y_o]->data());
        }

        ~LabelEncoder() {}
    };
   
}
#endif

