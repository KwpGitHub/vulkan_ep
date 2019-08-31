#ifndef CAST_H
#define CAST_H 

#include "../layer.h"

/*

The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.

Casting from string tensor in plain (e.g., "3.14" and "1000") and scientific numeric representations
(e.g., "1e-5" and "1E8") to float types is supported. For example, converting string "100.5" to an integer may
result 100. There are some string literals reserved for special floating-point values;
"+INF" (and "INF"), "-INF", and "NaN" are positive infinity, negative infinity, and not-a-number, respectively.
Any string which can exactly match "+INF" in a case-insensitive way would be mapped to positive infinite. Similarly,
this case-insensitive rule is applied to "INF" and "NaN". When casting from numeric tensors
to string tensors, plain floating-point representation (such as "314.15926") would be used. 
Converting non-numerical-literal string such as "Hello World!" is an undefined behavior. Cases 
of converting string representing floating-point arithmetic value, such as "2.718", to INT is an undefined behavior.

Conversion from a numerical type to any numerical type is always allowed.
User must be aware of precision loss and value change caused by range difference between two types.
For example, a 64-bit float 3.1415926459 may be round to a 32-bit float 3.141592. Similarly, converting
an integer 36 to Boolean may produce 1 because we truncate bits which can't be stored in the targeted type.

input: Input tensor to be cast.
output: Output tensor with the same shape as input with type specified by the 'to' argument
*/

//Cast
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               to
//PARAMETER_TYPES:          int
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Cast : public backend::Layer {
        typedef struct {
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        int m_to;
        std::string m_input_i;
        
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        Cast(std::string name);
        
        virtual void forward();        
        virtual void init( int _to); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~Cast() {}
    };
   
}
#endif

