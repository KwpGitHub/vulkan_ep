#ifndef SIZE_H
#define SIZE_H 

#include "../layer.h"

/*

Takes a tensor as input and outputs a int64 scalar that equals to the total number of elements of the input tensor.

input: An input tensor.
output: Total number of elements of the input tensor
*/

//Size
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   size_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class Size : public backend::Layer {
        typedef struct {
            int t;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string m_data_i;
        
        std::string m_size_o;
        

        binding_descriptor   binding;
       

    public:
        Size(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _data_i, std::string _size_o); 
        virtual void build();

        ~Size() {}
    };
   
}
#endif

