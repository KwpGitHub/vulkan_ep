#ifndef COMPRESS_H
#define COMPRESS_H 

#include "../layer.h"

/*

    Selects slices from an input tensor along a given axis where condition evaluates to True for each axis index.
    In case axis is not provided, input is flattened before elements are selected.
    Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
    
input: Tensor of rank r >= 1.
input: Rank 1 tensor of booleans to indicate which slices or data elements to be selected. Its length can be less than the input length alone the axis or the flattened input size if axis is not specified. In such cases data slices or elements exceeding the condition length are discarded.
output: Tensor of rank r if axis is specified. Otherwise output is a Tensor of rank 1.
*/

//Compress
//INPUTS:                   input_i, condition_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis
//OPTIONAL_PARAMETERS_TYPE: int


//class stuff
namespace layers {   

    class Compress : public backend::Layer {
        typedef struct {
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        int m_axis;
        std::string m_input_i; std::string m_condition_i;
        
        std::string m_output_o;
        

        binding_descriptor   binding;
       

    public:
        Compress(std::string name);
        
        virtual void forward();        
        virtual void init( int _axis); 
        virtual void bind(std::string _input_i, std::string _condition_i, std::string _output_o); 
        virtual void build();

        ~Compress() {}
    };
   
}
#endif

