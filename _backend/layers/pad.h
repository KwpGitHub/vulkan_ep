#include "../layer.h"
#ifndef PAD_H
#define PAD_H 
/*

Given `data` tensor, pads, mode, and value.
Example:
  Insert 0 pads to the beginning of the second dimension.
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  pads = [0, 2, 0, 0]
  output = [
      [
          [0.0, 0.0, 1.0, 1.2],
          [0.0, 0.0, 2.3, 3.4],
          [0.0, 0.0, 4.5, 5.7],
      ],
  ]

input: Input tensor.
output: Tensor after padding.
//*/
//Pad
//INPUTS:                   data_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               pads
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      mode, value
//OPTIONAL_PARAMETERS_TYPE: int, float

//class stuff
namespace backend {   

    class Pad : public Layer {
        typedef struct {
            Shape_t pads; int mode; float value;
			
            Shape_t data_input;
            
            Shape_t output_output;
            
        } binding_descriptor;

        Shape_t pads; int mode; float value;
        std::string data_input;
        
        std::string output_output;
        

        binding_descriptor   binding;

        vuh::Device* _get_device();
        vuh::Program<Specs, binding_descriptor>* program;        

    public:
        Pad(std::string n);
    
        void forward() { program->run(); }
        
        void init( Shape_t _pads,  int _mode,  float _value); 
        void bind(std::string _data_input, std::string _output_output); 

        ~Pad() {}

    };
    
}

#endif

