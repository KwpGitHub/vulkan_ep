#ifndef BINARIZER_H
#define BINARIZER_H 

#include "../layer.h"

/*

    Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.

input: Data to be binarized
output: Binarized output data
*/

//Binarizer
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      threshold
//OPTIONAL_PARAMETERS_TYPE: float


//class stuff
namespace layers {   

    class Binarizer : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        float threshold;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
       

    public:
        Binarizer(std::string name);
        
        virtual void forward();        
        virtual void init( float _threshold); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~Binarizer() {}
    };
   
}
#endif

