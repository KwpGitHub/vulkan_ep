#ifndef UNSQUEEZE_H
#define UNSQUEEZE_H 

#include "../layer.h"

/*

Insert single-dimensional entries to the shape of a tensor.
Takes one required argument `axes`, a list of dimensions that will be inserted.
Dimension indices in `axes` are as seen in the output tensor. For example:
  Given a tensor such that tensor with shape [3, 4, 5], then
  Unsqueeze(tensor, axes=[0, 4]) has shape [1, 3, 4, 5, 1]

input: Original tensor
output: Reshaped tensor with same data as input.
*/

//Unsqueeze
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   expanded_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               axes
//PARAMETER_TYPES:          std::vector<int>
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Unsqueeze : public backend::Layer {
        typedef struct {          
            backend::Shape_t data_i;
            
            backend::Shape_t expanded_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::vector<int> axes;
        std::string data_i;
        
        std::string expanded_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Unsqueeze(std::string name);
    
        void forward() { program->run(); }
        
        virtual void init( std::vector<int> _axes); 
        virtual void bind(std::string _data_i, std::string _expanded_o); 
        virtual void build();

        ~Unsqueeze() {}
    };
   
}
#endif

