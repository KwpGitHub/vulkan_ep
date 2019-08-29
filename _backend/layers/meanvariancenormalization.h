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
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<int> axes;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;
       

    public:
        MeanVarianceNormalization(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<int> _axes); 
        virtual void bind(std::string _X_i, std::string _Y_o); 
        virtual void build();

        ~MeanVarianceNormalization() {}
    };
   
}
#endif

