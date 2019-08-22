#ifndef REDUCELOGSUMEXP_H
#define REDUCELOGSUMEXP_H 

#include "../layer.h"

/*

Computes the log sum exponent of the input tensor's element along the provided axes. The resulted
tensor has the same rank as the input if keepdims equal 1. If keepdims equal 0, then
the resulted tensor have the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy default keepdims to
False instead of True.
input: An input tensor.
output: Reduced output tensor.
*/

//ReduceLogSumExp
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

    class ReduceLogSumExp : public backend::Layer {
        typedef struct {          
            backend::Shape_t data_i;
            
            backend::Shape_t reduced_o;
            
        } binding_descriptor;
        using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;

        std::vector<int> axes; int keepdims;
        std::string data_i;
        
        std::string reduced_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        ReduceLogSumExp(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( std::vector<int> _axes,  int _keepdims); 
        virtual void bind(std::string _data_i, std::string _reduced_o); 
        virtual void build();

        ~ReduceLogSumExp() {}
    };
   
}
#endif

