#ifndef PIPELINE_H
#define PIPELINE_H

#include "mat.h"
#include "utils.h"
#include <vulkan/vulkan.h>
#include "device.h"

namespace backend {

	class Option;
	class Pipeline
	{
	public:
		Pipeline(const VulkanDevice* vkdev);
		~Pipeline();

		void set_optimal_local_size_xyz(int w = 32, int h = 32, int c = 32);
		void set_local_size_xyz(int w, int h, int c);

		int create(const uint32_t* spv_data, size_t spv_data_size, const char* entry_name, const std::vector<vk_specialization_type>& specializations, int binding_count, int push_constant_count);
		int create(VkShaderModule shader_module, const char* entry_name, const std::vector<vk_specialization_type>& specializations, int binding_count, int push_constant_count);
		int create(const char* name, const Option& opt, const std::vector<vk_specialization_type>& specializations, int binding_count, int push_constant_count);
		void destroy();

		const VulkanDevice* vkdev;

		VkShaderModule local_shader_module;
		VkDescriptorSetLayout descriptorset_layout;
		VkPipelineLayout pipeline_layout;
		VkPipeline pipeline;
		VkDescriptorUpdateTemplateKHR descriptor_update_template;

		uint32_t local_size_x;
		uint32_t local_size_y;
		uint32_t local_size_z;

	protected:
		int create_descriptorset_layout(int binding_count);
		int create_pipeline_layout(int push_constant_count);
		int create_pipeline(VkShaderModule shader_module, const char* entry_name, const std::vector<vk_specialization_type>& specializations);
		int create_descriptor_update_template(int binding_count);


	};

}

#endif // !PIPELINE_H

