#ifndef MAXUNPOOL_H
#define MAXUNPOOL_H //MaxUnpool

//INPUTS:                   X_input, I_input
//OPTIONAL_INPUTS:          output_shape_input_o
//OUTPUS:                   output_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               kernel_shape
//PARAMETER_TYPES:          Shape_t
//OPTIONAL_PARAMETERS:      pads, strides
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Shape_t



namespace backend {
    class MaxUnpool : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t kernel_shape; Shape_t pads; Shape_t strides;
			
            //input
            Shape_t X_input; Shape_t I_input;
            Shape_t output_shape_input_o;
            //output
            Shape_t output_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        MaxUnpool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        Shape_t kernel_shape; Shape_t pads; Shape_t strides;
		
        //input
        std::string X_input; std::string I_input;
        std::string output_shape_input_o;
        //output
        std::string output_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~MaxUnpool(){}
    };
}


namespace backend {    
    MaxUnpool::MaxUnpool(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/maxunpool.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({kernel_shape, pads, strides, tensor_dict[X_input]->shape(), tensor_dict[I_input]->shape(), tensor_dict[output_shape_input_o]->shape(), tensor_dict[output_output]->shape()}, 
                            tensor_dict[X_input], tensor_dict[I_input], tensor_dict[output_shape_input_o],
                            tensor_dict[output_output] );
    }

    vuh::Device* MaxUnpool::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
