#ifndef GLOBALLPPOOL_H
#define GLOBALLPPOOL_H 

#include "../layer.h"

#include <pybind11/pybind11.h>
namespace py = pybind11;

/*

 GlobalLpPool consumes an input tensor X and applies lp pool pooling across
 the values in the same channel. This is equivalent to LpPool with kernel size
 equal to the spatial dimension of input tensor.
input: Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.
output: Output data tensor from pooling across the input tensor. Dimensions will be N x C x 1 x 1
*/

//GlobalLpPool
//INPUTS:                   X_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      p
//OPTIONAL_PARAMETERS_TYPE: int

//class stuff
namespace backend {   

    class GlobalLpPool : public Layer {
        typedef struct {
            int p;
			
            Shape_t X_i;
            
            Shape_t Y_o;
            
        } binding_descriptor;

        int p;
        std::string X_i;
        
        std::string Y_o;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        GlobalLpPool(const std::string& name);
    
        void forward() { program->run(); }
        
        void init( int _p); 
        void bind(std::string _X_i, std::string _Y_o); 

        ~GlobalLpPool() {}
    };

}

#endif

