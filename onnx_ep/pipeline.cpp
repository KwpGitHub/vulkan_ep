#include "pipeline.h"
#include <cstdio>
#include <math.h>
#include <algorithm>
#include <string>


namespace backend {
	Pipeline::Pipeline(const Device* _dev): dev(_dev) {
		local_shader_module = 0;
		descriptorset_layout = 0;
		pipeline_layout = 0;
		pipeline = 0;
		descriptor_update_template = 0;
	
		local_size_x = 1;
		local_size_y = 1;
		local_size_z = 1;
	}

	Pipeline::~Pipeline() { destroy(); }
	
	int Pipeline::create(const uint32_t* spv_data, size_t spv_data_size, const char* entry_name, const std::vector<vk_specialization_type>& specializations, int binding_count, int push_constant_count) {
		local_shader_module = dev->compile_shader_module(spv_data, spv_data_size);
		return create(local_shader_module, entry_name, specializations, binding_count, push_constant_count);
	}


	int Pipeline::create(VkShaderModule shader_module, const char* entry_name, const std::vector<vk_specialization_type>& specializations, int binding_count, int push_constant_count) {
		create_descriptorset_layout(binding_count);
		create_pipeline_layout(push_constant_count);
		create_pipeline(shader_module, entry_name, specializations);

		if (dev->info.support_VK_KHR_descriptor_update_template)
			create_descriptor_update_template(binding_count);

		return 0;
	}

	void Pipeline::destroy() {
		if (dev->info.support_VK_KHR_descriptor_update_template) {
			if (descriptorset_layout) {
				dev->vkDestroyDescriptorUpdateTemplateKHR(dev->vkdevice(), descriptor_update_template, 0);
				descriptor_update_template = 0;
			}
		}

		if (pipeline) {
			vkDestroyPipeline(dev->vkdevice(), pipeline, 0);
			pipeline = 0;
		}

		if (pipeline_layout) {
			vkDestroyPipelineLayout(dev->vkdevice(), pipeline_layout, 0);
			pipeline_layout = 0;
		}

		if (descriptorset_layout) {
			vkDestroyDescriptorSetLayout(dev->vkdevice(), descriptorset_layout, 0);
			descriptorset_layout = 0;
		}

		if (local_shader_module) {
			vkDestroyShaderModule(dev->vkdevice(), local_shader_module, 0);
			local_shader_module = 0;
		}

	}


	void Pipeline::set_optimal_local_size_xyz(int w, int h, int c)
	{
		if (c > 0) {
			local_size_z = dev->info.max_workgroup_size[2];
			while ((uint32_t)c < local_size_z)
				local_size_z /= 2;
		}
		else
			local_size_z = std::min<uint32_t>((uint32_t)128, dev->info.max_workgroup_size[2]);

		uint32_t max_local_size_xy = dev->info.max_workgroup_invocations / local_size_z;

		if (h == w || (h < 0 && w < 0)) {
			uint32_t local_size_xy = sqrt(max_local_size_xy);
			uint32_t local_size_xy_prefer = 128;
			while (local_size_xy < local_size_xy_prefer)
				local_size_xy_prefer /= 2;
			local_size_x = local_size_xy_prefer;
			local_size_y = local_size_xy_prefer;
		}
		if (h > 0 && w > 0) {
			if (h > w) {
				float ps = h / (float)w;
				float local_size_xy = sqrt(max_local_size_xy / ps);
				local_size_y = local_size_xy * ps;
				local_size_x = std::max<uint32_t>((uint32_t)local_size_xy, (uint32_t)1);
			}
			else {
				float ps = w / (float)h;
				float local_size_xy = sqrt(max_local_size_xy / ps);
				local_size_y = std::max<uint32_t>((uint32_t)local_size_xy, (uint32_t)1);
				local_size_x = local_size_xy * ps;
			}

			uint32_t local_size_y_prefer = std::min<uint32_t>((uint32_t)128, dev->info.max_workgroup_size[1]);
			while (local_size_y < local_size_y_prefer)
				local_size_y_prefer /= 2;

			uint32_t local_size_x_prefer = std::min<uint32_t>((uint32_t)128, dev->info.max_workgroup_size[0]);
			while (local_size_x < local_size_x_prefer)
				local_size_x_prefer /= 2;
	
			local_size_y = local_size_y_prefer;
			local_size_x = local_size_x_prefer;
		}
		else if (h > 0) {
			local_size_y = std::min<uint32_t>(max_local_size_xy, dev->info.max_workgroup_size[1]);
			while ((uint32_t)h < local_size_y)
				local_size_y /= 2;

			uint32_t max_local_size_x = max_local_size_xy / local_size_y;
			local_size_x = std::min<uint32_t>(max_local_size_x, dev->info.max_workgroup_size[0]);
		}
		else if (w > 0) {
			local_size_x = std::min<uint32_t>(max_local_size_xy, dev->info.max_workgroup_size[0]);
			while ((uint32_t)w < local_size_x)
				local_size_x /= 2;
		
			uint32_t max_local_size_y = max_local_size_xy / local_size_x;
			local_size_y = std::min<uint32_t>(max_local_size_y, dev->info.max_workgroup_size[1]);
		}

	}

	void Pipeline::set_local_size_xyz(int w, int h, int c) { local_size_x = w; local_size_y = h; local_size_z = c; }

	int Pipeline::create_descriptorset_layout(int binding_count) {
		if (binding_count == 0) {
			descriptorset_layout = 0;
			return 0;
		}

		std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindings(binding_count);
		for (int i = 0; i < binding_count; ++i) {
			descriptorSetLayoutBindings[i].binding = i;
			descriptorSetLayoutBindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorSetLayoutBindings[i].descriptorCount = 1;
			descriptorSetLayoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
			descriptorSetLayoutBindings[i].pImmutableSamplers = 0;
		}

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo;
		descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptorSetLayoutCreateInfo.pNext = 0;
		descriptorSetLayoutCreateInfo.flags = 0;
		descriptorSetLayoutCreateInfo.bindingCount = binding_count;
		descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings.data();

		if (dev->info.support_VK_KHR_push_descriptor)
			descriptorSetLayoutCreateInfo.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;

		VkResult ret = vkCreateDescriptorSetLayout(dev->vkdevice(), &descriptorSetLayoutCreateInfo, 0, &descriptorset_layout);
		if (ret != VK_SUCCESS) return -1;
		return 0;
	}

	int Pipeline::create_pipeline_layout(int push_constant_count) {
		VkPushConstantRange pushConstantRange;
		pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(int) * push_constant_count;

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
		pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutCreateInfo.pNext = 0;
		pipelineLayoutCreateInfo.flags = 0;

		if (descriptorset_layout) {
			pipelineLayoutCreateInfo.setLayoutCount = 1;
			pipelineLayoutCreateInfo.pSetLayouts = &descriptorset_layout;
		}
		else {
			pipelineLayoutCreateInfo.setLayoutCount = 0;
			pipelineLayoutCreateInfo.pSetLayouts = 0;
		}

		if (push_constant_count > 0) {
			pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
			pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
		}
		else {
			pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
			pipelineLayoutCreateInfo.pPushConstantRanges = 0;
		}

		VkResult ret = vkCreatePipelineLayout(dev->vkdevice(), &pipelineLayoutCreateInfo, 0, &pipeline_layout);
		if (ret != VK_SUCCESS) return -1;
		return 0;
	}

	int Pipeline::create_pipeline(VkShaderModule shader_module, const char* entry_name, const std::vector<vk_specialization_type>& specializations) {
		const int specialization_count = specializations.size();
		std::vector<VkSpecializationMapEntry> specializationMapEntries;
		specializationMapEntries.resize(specialization_count + 3);
		for (int i = 0; i < specialization_count; ++i) {
			specializationMapEntries[i].constantID = i;
			specializationMapEntries[i].offset = i * sizeof(vk_specialization_type);
			specializationMapEntries[i].size = sizeof(vk_specialization_type);
		}

		std::vector<vk_specialization_type> specialization_data = specializations;
		{
			VkSpecializationMapEntry* local_size_xyz_entries = specializationMapEntries.data() + specialization_count;

			local_size_xyz_entries[0].constantID = 233;
			local_size_xyz_entries[0].offset = (specialization_count + 0) * sizeof(vk_specialization_type);
			local_size_xyz_entries[0].size = sizeof(vk_specialization_type);

			local_size_xyz_entries[1].constantID = 234;
			local_size_xyz_entries[1].offset = (specialization_count + 1) * sizeof(vk_specialization_type);
			local_size_xyz_entries[1].size = sizeof(vk_specialization_type);

			local_size_xyz_entries[2].constantID = 235;
			local_size_xyz_entries[2].offset = (specialization_count + 2) * sizeof(vk_specialization_type);
			local_size_xyz_entries[2].size = sizeof(vk_specialization_type);

			specialization_data.resize(specialization_count + 3);
			specialization_data[specialization_count + 0].u32 = local_size_x;
			specialization_data[specialization_count + 1].u32 = local_size_y;
			specialization_data[specialization_count + 2].u32 = local_size_z;
		}


		VkSpecializationInfo specializationInfo;
		specializationInfo.mapEntryCount = specializationMapEntries.size();
		specializationInfo.pMapEntries = specializationMapEntries.data();
		specializationInfo.dataSize = specialization_data.size() * sizeof(vk_specialization_type);
		specializationInfo.pData = specialization_data.data();

		VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo;
		pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		pipelineShaderStageCreateInfo.pNext = 0;
		pipelineShaderStageCreateInfo.flags = 0;
		pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		pipelineShaderStageCreateInfo.module = shader_module;
		pipelineShaderStageCreateInfo.pName = entry_name;
		pipelineShaderStageCreateInfo.pSpecializationInfo = &specializationInfo;

		VkComputePipelineCreateInfo computePipelineCreateInfo;
		computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		computePipelineCreateInfo.pNext = 0;
		computePipelineCreateInfo.flags = 0;
		computePipelineCreateInfo.stage = pipelineShaderStageCreateInfo;
		computePipelineCreateInfo.layout = pipeline_layout;
		computePipelineCreateInfo.basePipelineHandle = 0;
		computePipelineCreateInfo.basePipelineIndex = 0;

		VkResult ret = vkCreateComputePipelines(dev->vkdevice(), 0, 1, &computePipelineCreateInfo, 0, &pipeline);
		if (ret != VK_SUCCESS) return -1;
		return 0;	
	}


	int Pipeline::create_descriptor_update_template(int binding_count)
	{
		if (binding_count == 0)
			descriptor_update_template = 0;
			return 0;

		std::vector<VkDescriptorUpdateTemplateEntryKHR> descriptorUpdateTemplateEntries(binding_count);
		for (int i = 0; i < binding_count; ++i)	{ // TODO do not update weights
			descriptorUpdateTemplateEntries[i].dstBinding = i;
			descriptorUpdateTemplateEntries[i].dstArrayElement = 0;
			descriptorUpdateTemplateEntries[i].descriptorCount = 1;
			descriptorUpdateTemplateEntries[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorUpdateTemplateEntries[i].offset = i * sizeof(VkDescriptorBufferInfo);
			descriptorUpdateTemplateEntries[i].stride = sizeof(VkDescriptorBufferInfo);
		}

		VkDescriptorUpdateTemplateCreateInfoKHR descriptorUpdateTemplateCreateInfo;
		descriptorUpdateTemplateCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR;
		descriptorUpdateTemplateCreateInfo.pNext = 0;
		descriptorUpdateTemplateCreateInfo.flags = 0;
		descriptorUpdateTemplateCreateInfo.descriptorUpdateEntryCount = binding_count;// TODO do not update weights
		descriptorUpdateTemplateCreateInfo.pDescriptorUpdateEntries = descriptorUpdateTemplateEntries.data();
		if (dev->info.support_VK_KHR_push_descriptor)
			descriptorUpdateTemplateCreateInfo.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
		else
			descriptorUpdateTemplateCreateInfo.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET_KHR;
		// descriptorSetLayout should be ignored if VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR
		// FIXME HACK WARNING TODO NOTE but crash on radv if set NULL  :(
		descriptorUpdateTemplateCreateInfo.descriptorSetLayout = descriptorset_layout;
		descriptorUpdateTemplateCreateInfo.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
		descriptorUpdateTemplateCreateInfo.pipelineLayout = pipeline_layout;
		descriptorUpdateTemplateCreateInfo.set = 0;

		VkResult ret = dev->vkCreateDescriptorUpdateTemplateKHR(dev->vkdevice(), &descriptorUpdateTemplateCreateInfo, 0, &descriptor_update_template);
		if (ret != VK_SUCCESS) return -1;
		return 0;
	}

}