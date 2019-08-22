#ifndef SPLIT_H
#define SPLIT_H 

#include "../layer.h"

/*
Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.

input: The tensor to split
output: One or more outputs forming list of tensors after splitting
*/

//Split
//INPUTS:                   input_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axis, split
//OPTIONAL_PARAMETERS_TYPE: int, std::vector<int>


//class stuff
namespace layers {   

    class Split : public backend::Layer {
        typedef struct {          
            backend::Shape_t input_i;
            
            
            
        } binding_descriptor;
        using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;

        int axis; std::vector<int> split;
        std::string input_i;
        
        
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Split(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( int _axis,  std::vector<int> _split); 
        virtual void bind(std::string _input_i); 
        virtual void build();

        ~Split() {}
    };
   
}
#endif

