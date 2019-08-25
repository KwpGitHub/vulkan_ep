#ifndef MEANVARIANCENORMALIZATION_H
#define MEANVARIANCENORMALIZATION_H 

#include "../layer.h"

/*

      A MeanVarianceNormalization Function: Perform mean variance normalization
      on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ```

input: Input tensor
output: Output tensor
*/

//MeanVarianceNormalization
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes
//OPTIONAL_PARAMETERS_TYPE: std::vector<int>


//class stuff
namespace layers {   

    class MeanVarianceNormalization : public backend::Layer {
        typedef struct {          
            backend::Shape_t X_i;
            
            backend::Shape_t Y_o;
            
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::vector<int> axes;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
        vuh::Device* _get_device();

        /*using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	    struct Params { uint32_t size; float a; };    // shader push-constants interface
	    vuh::Program<Specs, Params>* program;*/


    public:
        MeanVarianceNormalization(std::string name);
        
        void forward() { program->run(); }
        
        virtual void init( std::vector<int> _axes); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~MeanVarianceNormalization() {}
    };
   
}
#endif

