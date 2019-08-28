#ifndef SQUEEZE_H
#define SQUEEZE_H 

#include "../layer.h"

/*

Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.

input: Tensors with at least max(dims) dimensions.
output: Reshaped tensor with same data as input.
*/

//Squeeze
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   squeezed_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes
//OPTIONAL_PARAMETERS_TYPE: std::vector<int>


//class stuff
namespace layers {   

    class Squeeze : public backend::Layer {
        typedef struct {
            uint32_t size; float a;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<int> axes;
        std::string data_i;
        
        std::string squeezed_o;
        

        binding_descriptor   binding;
       

    public:
        Squeeze(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<int> _axes); 
        virtual void bind(std::string _data_i, std::string _squeezed_o); 
        virtual void build();

        ~Squeeze() {}
    };
   
}
#endif

