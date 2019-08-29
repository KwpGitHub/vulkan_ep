#ifndef PAD_H
#define PAD_H 

#include "../layer.h"

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
*/

//Pad
//INPUTS:                   data_i
//OPTIONAL_INPUTS:          
//OUTPUS:                   output_o
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               pads
//PARAMETER_TYPES:          std::vector<int>
//OPTIONAL_PARAMETERS:      mode, value
//OPTIONAL_PARAMETERS_TYPE: std::string, float


//class stuff
namespace layers {   

    class Pad : public backend::Layer {
        typedef struct {
            uint32_t size;
        } binding_descriptor;
        
        vuh::Program<Specs, binding_descriptor>* program;
        std::string file;        
		vuh::Device* dev;
        std::vector<backend::Shape_t> SHAPES;
        vuh::Array<backend::Shape_t>* _SHAPES;

        std::vector<int> pads; std::string mode; float value;
        std::string data_i;
        
        std::string output_o;
        

        binding_descriptor   binding;
       

    public:
        Pad(std::string name);
        
        virtual void forward();        
        virtual void init( std::vector<int> _pads,  std::string _mode,  float _value); 
        virtual void bind(std::string _data_i, std::string _output_o); 
        virtual void build();

        ~Pad() {}
    };
   
}
#endif

