#ifndef LABELENCODER_H
#define LABELENCODER_H 

#include "../layer.h"

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
//OPTIONAL_PARAMETERS_TYPE: float, int, std::string, std::vector<float>, std::vector<int>, std::vector<std::string>, std::vector<float>, std::vector<int>, std::vector<std::string>


//class stuff
namespace layers {   

    class LabelEncoder : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        float m_default_float; int m_default_int64; std::string m_default_string; std::vector<float> m_keys_floats; std::vector<int> m_keys_int64s; std::vector<std::string> m_keys_strings; std::vector<float> m_values_floats; std::vector<int> m_values_int64s; std::vector<std::string> m_values_strings;
        std::string m_X_i;
        
        std::string m_Y_o;
        

        binding_descriptor   binding;
       

    public:
        LabelEncoder(std::string name);
        
        virtual void forward();        
        virtual void init( float _default_float,  int _default_int64,  std::string _default_string,  std::vector<float> _keys_floats,  std::vector<int> _keys_int64s,  std::vector<std::string> _keys_strings,  std::vector<float> _values_floats,  std::vector<int> _values_int64s,  std::vector<std::string> _values_strings); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~LabelEncoder() {}
    };
   
}
#endif

