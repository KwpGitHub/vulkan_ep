#ifndef ROIALIGN_H
#define ROIALIGN_H //RoiAlign

//INPUTS:                   X, rois, batch_indices
//OPTIONAL_INPUTS:          
//OUTPUS:                   Y
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      mode, output_height, output_width, sampling_ratio, spatial_scale
//OPTIONAL_PARAMETERS_TYPE: int, int, int, int, float



namespace backend {
    class RoiAlign : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            int mode; int output_height; int output_width; int sampling_ratio; float spatial_scale;
			
            //input
            Shape_t X; Shape_t rois; Shape_t batch_indices;
            
            //output
            Shape_t Y;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        RoiAlign(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward(){ program->run(); }
        
        int mode; int output_height; int output_width; int sampling_ratio; float spatial_scale;
		
        //input
        std::string X; std::string rois; std::string batch_indices;
        
        //output
        std::string Y;
        
        //std::vector<uint32_t> output_shape();
   
        ~RoiAlign(){}
    };
}


namespace backend {    
    RoiAlign::RoiAlign(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/roialign.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            //program->bind({}, );
    }

    vuh::Device* RoiAlign::_get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
            }
            return device;
    }
};

#endif
