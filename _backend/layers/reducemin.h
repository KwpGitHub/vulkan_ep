#ifndef REDUCEMIN_H
#define REDUCEMIN_H 

#include "../layer.h"

/*

Computes the min of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
input: An input tensor.
output: Reduced output tensor.
*/

//ReduceMin
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   reduced_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes, keepdims
//OPTIONAL_PARAMETERS_TYPE: std::vector<int>, int


//class stuff
namespace layers {   

    class ReduceMin : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<int> axes; int keepdims;
        std::string data_i;
        
        std::string reduced_o;
        

        binding_descriptor   binding;
       

    public:
        ReduceMin(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<int> _axes,  int _keepdims); 
        virtual void bind(std::string _data_i, std::string _reduced_o); 
        virtual void build();

        ~ReduceMin() {}
    };
   
}
#endif

