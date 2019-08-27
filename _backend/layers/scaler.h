#ifndef SCALER_H
#define SCALER_H 

#include "../layer.h"

/*

    Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.

input: Data to be scaled.
output: Scaled output data.
*/

//Scaler
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      offset, scale
//OPTIONAL_PARAMETERS_TYPE: std::vector<float>, std::vector<float>


//class stuff
namespace layers {   

    class Scaler : public backend::Layer {
        typedef struct {          
            backend::Shape_t X_i;
            
            backend::Shape_t Y_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::vector<float> offset; std::vector<float> scale;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        Scaler(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<float> _offset,  std::vector<float> _scale); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~Scaler() {}
    };
   
}
#endif

