#ifndef COMMAND_H
#define COMMAND_H

#include "pipeline.h"
#include <vector>
#include <vulkan/vulkan.h>
#include "mat.h"
#include "utils.h"

namespace backend {

	class Command
	{
	public:
		Command(const VulkanDevice* vkdev, uint32_t queue_family_index);
		virtual ~Command();

	protected:
		int create_command_pool();
		int create_command_buffer();
		int begin_command_buffer();
		int end_command_buffer();
		int queue_submit_and_wait_fence();

		const VulkanDevice* vkdev;
		uint32_t queue_family_index;
		VkCommandPool command_pool;
		VkCommandBuffer command_buffer;
		VkFence fence;
	};


	class VkCompute : public Command {
	public:
		VkCompute(const VulkanDevice* vkdev);
		~VkCompute();

		void record_upload(const VkMat& m);
		void record_download(const VkMat& m);
		void record_clone(const VkMat& src, const VkMat& dst);
		void record_copy_region(const VkMat& src, const VkMat& dst, const VkBufferCopy& region);
		void record_copy_regions(const VkMat& src, const VkMat& dst, const std::vector<VkBufferCopy>& regions);
		void record_pipeline(const Pipeline* pipeline, const std::vector<VkMat>& bindings, const std::vector<vk_constant_type>& constants, const VkMat& m);
		//void record_write_timestamp(uint32_t query);
		int submit_and_wait();
		int reset();

	protected:
		void record_bind_pipeline(VkPipeline pipeline);
		void record_update_bindings(VkPipelineLayout pipeline_layout, VkDescriptorSetLayout descriptorset_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const std::vector<VkMat>& bindings);
		void record_push_constants(VkPipelineLayout pipeline_layout, const std::vector<vk_constant_type>& constants);
		void record_dispatch(const uint32_t* group_count_xyz);
		void record_transfer_compute_barrier(const VkMat& m);
		void record_compute_transfer_barrier(const VkMat& m);
		void record_compute_compute_barrier(const VkMat& m);
		void record_transfer_transfer_barrier(const VkMat& m);
		void record_prepare_transfer_barrier(const VkMat& m);
		void record_prepare_compute_barrier(const VkMat& m);

		void copy_buffer(VkBuffer src, size_t src_offset, VkBuffer dst, size_t dst_offset, size_t size);
		void copy_buffer_regions(VkBuffer src, VkBuffer dst, const std::vector<VkBufferCopy>& regions);
		void bind_pipeline(VkPipeline pipeline);
		void bind_descriptorset(VkPipelineLayout pipeline_layout, VkDescriptorSet descriptorset);
		void update_bindings(VkPipelineLayout pipeline_layout, VkDescriptorUpdateTemplateKHR descriptor_update_template, const std::vector<VkDescriptorBufferInfo>& descriptorBufferInfos);
		void push_constants(VkPipelineLayout pipeline_layout, const std::vector<vk_constant_type>& constants);
		void dispatch(const uint32_t* group_count_xyz);
		void transfer_compute_barrier(VkBuffer buffer, size_t offset, size_t size);
		void compute_transfer_barrier(VkBuffer buffer, size_t offset, size_t size);
		void compute_compute_barrier(VkBuffer buffer, size_t offset, size_t size);
		void transfer_transfer_barrier(VkBuffer buffer, size_t offset, size_t size);

		std::vector<VkDescriptorPool> descriptor_pools;
		std::vector<VkDescriptorSet> descriptorsets;
		struct record_type
		{
			int type;
			union
			{
				struct { VkBuffer src; size_t src_offset; VkBuffer dst; size_t dst_offset; size_t size; } copy;
				struct { VkBuffer src; VkBuffer dst; } copy_regions;
				struct { VkPipeline pipeline; } bind_pipeline;
				struct { VkPipelineLayout pipeline_layout; VkDescriptorSet descriptorset; } bind_descriptorset;
				struct { VkPipelineLayout pipeline_layout; } push_constants;
				struct { uint32_t group_count_xyz[3]; } dispatch;
				struct { VkBuffer buffer; size_t offset; size_t size; } transfer_compute_barrier;
				struct { VkBuffer buffer; size_t offset; size_t size; } compute_transfer_barrier;
				struct { VkBuffer buffer; size_t offset; size_t size; } compute_compute_barrier;
				struct { VkBuffer buffer; size_t offset; size_t size; } transfer_transfer_barrier;
			};
			std::vector<VkBufferCopy> regions;
			std::vector<vk_constant_type> constants;
		};
		std::vector<record_type> delayed_records;
	};

	class VkTransfer : public Command {
	public:
		VkTransfer(const VulkanDevice* vkdev);
		~VkTransfer();
		void record_upload(const Mat& src, VkMat& dst, const Option& opt);
		int submit_and_wait();
		VkAllocator* weight_vkallocator;
		VkAllocator* staging_vkallocator;

	protected:
		void copy_buffer(VkBuffer src, size_t src_offset, VkBuffer dst, size_t dst_offset, size_t size);
		void copy_buffer_regions(VkBuffer src, VkBuffer dst, const std::vector<VkBufferCopy>& regions);

		size_t buffer_offset_alignment;
		VkBufferMemory* staging_data;
		struct record_type
		{
			size_t size;
			Mat mat;
			VkMat vkmat;
		};
		std::vector<record_type> delayed_records;
	};

}

#endif //!COMAND_H