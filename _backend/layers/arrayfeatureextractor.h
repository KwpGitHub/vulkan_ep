#ifndef ARRAYFEATUREEXTRACTOR_H
#define ARRAYFEATUREEXTRACTOR_H 

#include "../layer.h"

/*

    Select elements of the input tensor based on the indices passed.<br>
    The indices are applied to the last axes of the tensor.

input: Data to be selected
input: The indices, based on 0 as the first index of any dimension.
output: Selected output data as an array
*/

//ArrayFeatureExtractor
//INPUTS:                   X_i, Y_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      
//OPTIONAL_PARAMETERS_TYPE: 


//class stuff
namespace layers {   

    class ArrayFeatureExtractor : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        
        std::string m_X_i; std::string m_Y_i;
        
        std::string m_Z_o;
        

        binding_descriptor   binding;
       

    public:
        ArrayFeatureExtractor(std::string name);
        
        virtual void forward();        
        virtual void init(); 
        virtual void bind(std::string _X_i, std::string _Y_i, std::string _Z_o); 
        virtual void build();

        ~ArrayFeatureExtractor() {}
    };
   
}
#endif

