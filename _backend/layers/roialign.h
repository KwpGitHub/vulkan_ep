#ifndef ROIALIGN_H
#define ROIALIGN_H

#include <vector>
#include "../layer.h"
#include "../kernel/vuh.h"

namespace backend {
    class RoiAlign : public Layer {
        struct Params{Shape_t X_t; Shape_t rois_t; Shape_t batch_indices_t; Shape_t Y_t; int mode_t; int output_height_t; int output_width_t; int sampling_ratio_t; float spatial_scale_t;
        };
            
        vuh::Program<Specs, Params>* program;

        vuh::Device* _get_device() {
            for(auto t_name: inputs) {
                if(tensor_dict.end() != tensor_dict.find(t_name)) 
                    return tensor_dict[t_name]->dev;
            }
            return device;
        }

        std::string X; std::string rois; std::string batch_indices; std::string Y;
        //parameter 
        Shape_t X_t; Shape_t rois_t; Shape_t batch_indices_t; Shape_t Y_t; int mode_t; int output_height_t; int output_width_t; int sampling_ratio_t; float spatial_scale_t;

    public:
        RoiAlign(std::string n, std::vector<std::string> i, std::vector<std::string> o, std::map<std::string, std::vector<std::string>> a): Layer(n, i, o, a) {
            X = i[0]; rois = i[1]; batch_indices = i[2];
            Y = o[0];
            program = new vuh::Program<Specs, Params>(*_get_device(), (file_path + std::string("\shaders/bin/roialign.spv")).c_str());
            program->grid(1024/PROCESSKERNEL_SIZE, 1024/PROCESSKERNEL_SIZE, 64/PROCESSKERNEL_SIZE);
			program->spec(64,64,64);
            program->bind({X_t, rois_t, batch_indices_t, Y_t, mode_t, output_height_t, output_width_t, sampling_ratio_t, spatial_scale_t }, tensor_dict[Y], tensor_dict[X], tensor_dict[rois], tensor_dict[batch_indices]);

        }
        
        void parameter_proc(std::map<std::string, std::vector<std::string>> a){
            convert_vec_param(a["X"], X_t);
			convert_vec_param(a["rois"], rois_t);
			convert_vec_param(a["batch_indices"], batch_indices_t);
			convert_vec_param(a["Y"], Y_t);
			convert_vec_param(a["mode"], mode_t);
			convert_vec_param(a["output_height"], output_height_t);
			convert_vec_param(a["output_width"], output_width_t);
			convert_vec_param(a["sampling_ratio"], sampling_ratio_t);
			convert_vec_param(a["spatial_scale"], spatial_scale_t);   
        }

        //Tensor* operator()(const Tensor* t) {            
        //}

		void forward(){
		}

       /* std::vector<uint32_t> output_shape(){
            for(auto t_name : inputs){
                if(tensor_dict.end() == tensor_dict.find(t_name) && layer_dict.end() != layer_dict.find(t_name)){
                    //need to do math
                    return layer_dict[t_name]->output_shape();
                }
                else if (tensor_dict.end() != tensor_dict.find(t_name) && layer_dict.end() == layer_dict.find(t_name)){
                    //need to do math
                    return tensor_dict[t_name]->dims;
                }

            }
            for(auto t_name : outputs){
                if(tensor_dict.end() != tensor_dict.find(t_name) && layer_dict.end() == layer_dict.find(t_name)){
                    return tensor_dict[t_name]->dims;
                }
            }
        }*/

    
        ~RoiAlign(){}

    };
}

#endif
