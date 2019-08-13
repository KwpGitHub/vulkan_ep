#ifndef ZIPMAP_H
#define ZIPMAP_H //ZipMap

#include "../layer.h"

//INPUTS:                   X_input
//OPTIONAL_INPUTS:          
//OUTPUS:                   Z_output
//OPTIONAL_OUTPUTS:         
//PARAMETERS:               
//PARAMETER_TYPES:          
//OPTIONAL_PARAMETERS:      classlabels_int64s, classlabels_strings
//OPTIONAL_PARAMETERS_TYPE: Shape_t, Tensor*



namespace backend {
    class ZipMap : public Layer {
        
        vuh::Device* _get_device();

        struct Params{
            Shape_t classlabels_int64s;
			Shape_t classlabels_strings;
            //input
            Shape_t X_input;
            
            //output
            Shape_t Z_output;
            
        };

        vuh::Program<Specs, Params>* program;

    public:
        ZipMap(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a);
        void forward() { program->run(); }
        
        Shape_t classlabels_int64s; Tensor* classlabels_strings;
		Shape_t classlabels_strings_s;
        //input
        std::string X_input;
        
        //output
        std::string Z_output;
        
        //std::vector<uint32_t> output_shape();
   
        ~ZipMap() {}
    };
}


namespace backend {    
    ZipMap::ZipMap(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a) : Layer(n, i, o, a) {            
        program = new vuh::Program<Specs, Params>(*_get_device(), std::string(file_path + "/shaders/bin/zipmap.spv").c_str());
        program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
        program->spec(64,64,64);
        program->bind({classlabels_int64s, classlabels_strings_s, tensor_dict[X_input]->shape(), tensor_dict[Z_output]->shape()} 
                        , *classlabels_strings
                        , tensor_dict[X_input], tensor_dict[Z_output] );
    }

    vuh::Device* ZipMap::_get_device() {
        for(auto t_name: inputs) {
            if(tensor_dict.end() != tensor_dict.find(t_name)) return tensor_dict[t_name]->dev;
        }
        return device;
    }
};

#endif
