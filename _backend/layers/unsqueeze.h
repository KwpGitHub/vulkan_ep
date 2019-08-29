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
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<int> axes;
        std::string data_i;
        
        std::string expanded_o;
        

        binding_descriptor   binding;
       

    public:
        Unsqueeze(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<int> _axes); 
        virtual void bind(std::string _data_i, std::string _expanded_o); 
        virtual void build();

        ~Unsqueeze() {}
    };
   
}
#endif

