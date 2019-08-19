#ifndef MEANVARIANCENORMALIZATION_H
#define MEANVARIANCENORMALIZATION_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

      A MeanVarianceNormalization Function: Perform mean variance normalization
      on the input tensor X using formula: <br/> ``` (X-EX)/sqrt(E(X-EX)^2) ```

input: Input tensor
output: Output tensor
*/

//MeanVarianceNormalization
//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      axes
//OPTIONAL_PARAMETERS_TYPE: Shape_t

//class stuff
namespace backend {   

    class MeanVarianceNormalization : public Layer {
        typedef struct {
            Shape_t axes;
			
            Shape_t X_input;
            
            Shape_t Y_output;
            
        } binding_descriptor;

        Shape_t axes;
        std::string X_input;
        
        std::string Y_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        MeanVarianceNormalization();
    
        void forward() { program->run(); }
        
        void init( Shape_t _axes); 
        void bind(std::string _X_input, std::string _Y_output); 

        ~MeanVarianceNormalization() {}
    };

    
    void init_layer_MeanVarianceNormalization(py::module& m) {
        // py::class_(m, "MeanVarianceNormalization");
    }
    

}


#endif

