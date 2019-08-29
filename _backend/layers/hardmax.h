#ifndef HARDMAX_H
#define HARDMAX_H 

#include "../layer.h"

/*

The operator computes the hardmax (1 for the first maximum value, and 0 for all others) values for each layer in the batch
 of the given input. The input is a 2-D tensor (Tensor<float>) of size
(batch_size x input_feature_dimensions). The output tensor has the same shape
and contains the hardmax values of the corresponding input.

Input does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
input \in [a_0, a_1, ..., a_{k-1}, a_k, ..., a_{n-1}] and k is
the axis provided, then input will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the input tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = N and a_1 * ... * a_{n-1} = D.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.

input: The input tensor that's coerced into a 2D matrix of size (NxD) as described above.
output: The output values with the same shape as input tensor (the original size without coercion).
*/

//Hardmax
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int


//class stuff
namespace layers {   

    class Hardmax : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        int axis;
        std::string input_i;
        
        std::string output_o;
        

        binding_descriptor   binding;
       

    public:
        Hardmax(std::string name);
        
        virtual void forward();        
        virtual void init( int _axis); 
        virtual void bind(std::string _input_i, std::string _output_o); 
        virtual void build();

        ~Hardmax() {}
    };
   
}
#endif

