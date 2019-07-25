#include "allocator.h"
#include <cstdio>
#include <algorithm>
#include "device.h"

namespace backend{
	CPU_Allocator::~CPU_Allocator()
	{

	}

	Allocator::Allocator(const Device* _dev) : dev(_dev){
		mappable = false;
	}
	
	VkBuffer Allocator::create_buffer(size_t size, VkBufferUsageFlags usage) {
		VkBufferCreateInfo bufferCreateInfo;
		bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferCreateInfo.pNext = 0;
		bufferCreateInfo.flags = 0;
		bufferCreateInfo.size = size;
		bufferCreateInfo.usage = usage;
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		bufferCreateInfo.queueFamilyIndexCount = 0;
		bufferCreateInfo.pQueueFamilyIndices = 0;

		VkBuffer buffer;
		VkResult ret = vkCreateBuffer(dev->vkdevice(), &bufferCreateInfo, 0, &buffer);
		if (ret != VK_SUCCESS) return 0;
		return buffer;
	}

	VkDeviceMemory Allocator::allocate_memory(size_t size, uint32_t memory_type_index) {
		VkMemoryAllocateInfo memoryAllocateInfo;
		memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memoryAllocateInfo.pNext = 0;
		memoryAllocateInfo.allocationSize = size;
		memoryAllocateInfo.memoryTypeIndex = memory_type_index;

		VkDeviceMemory memory = 0;
		VkResult ret = vkAllocateMemory(dev->vkdevice(), &memoryAllocateInfo, 0, &memory);
		if(ret != VK_SUCCESS) fprintf(stderr, "vkAllocateMemory failed %d\n", ret);
		return memory;
	}

	VkDeviceMemory Allocator::allocate_dedicated_memory(size_t size, uint32_t memory_type_index, VkBuffer buffer) {
		VkMemoryAllocateInfo memoryAllocateInfo;
		memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memoryAllocateInfo.pNext = 0;
		memoryAllocateInfo.allocationSize = size;
		memoryAllocateInfo.memoryTypeIndex = memory_type_index;

		VkMemoryDedicatedAllocateInfoKHR memoryDedicatedAllocateInfo;
		memoryDedicatedAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO_KHR;
		memoryDedicatedAllocateInfo.pNext = 0;
		memoryDedicatedAllocateInfo.image = 0;
		memoryDedicatedAllocateInfo.buffer = buffer;
		memoryAllocateInfo.pNext = &memoryDedicatedAllocateInfo;

		VkDeviceMemory memory = 0;
		VkResult ret = vkAllocateMemory(dev->vkdevice(), &memoryAllocateInfo, 0, &memory);
		if(ret != VK_SUCCESS) fprintf(stderr, "vkAllocateMemory failed %d\n", ret);
		return memory;
	}


	static inline size_t least_common_multiple(size_t a, size_t b) {
		if (a == b)	return a;
		if (a > b) return least_common_multiple(b, a);
		size_t lcm = b;
		while (lcm % a != 0) lcm += b;
		return lcm;
	}

	BlobBufferAllocator::BlobBufferAllocator(const Device* _dev) : Allocator(_dev) {
		mappable = dev->info.device_local_memory_index == dev->info.unified_memory_index;
		buffer_offset_alignment = dev->info.buffer_offset_alignment;
		if (mappable) {
			size_t memory_map_alignment = dev->info.memory_map_alignment;
			buffer_offset_alignment = least_common_multiple(buffer_offset_alignment, memory_map_alignment);
		}
		block_size = alignSize(16 * 1024 * 1024, buffer_offset_alignment); // 16,777,216‬
	}

	BlobBufferAllocator::~BlobBufferAllocator() { clear(); }
	
	void BlobBufferAllocator::set_block_size(size_t _blob_size) {
		block_size = _blob_size;
	}

	void BlobBufferAllocator::clear() {
		for (size_t i = 0; i < buffer_blocks.size(); ++i) {
			BufferMemory* ptr = buffer_blocks[i];
			if(mappable) vkUnmapMemory(dev->vkdevice(), ptr->memory);
			vkDestroyBuffer(dev->vkdevice(), ptr->buffer, 0);
			vkFreeMemory(dev->vkdevice(), ptr->memory, 0);
			delete ptr;
		}
		buffer_blocks.clear();
		budgets.clear();
	}

	BufferMemory* BlobBufferAllocator::fastMalloc(size_t size) {
		size_t aligned_size = alignSize(size, buffer_offset_alignment);
		const int buffer_block_count = buffer_blocks.size();

		for (int i = 0; i < buffer_block_count; ++i) {
			std::list< std::pair<size_t, size_t> >::iterator it = budgets[i].begin();
			while (it != budgets[i].end()) {
				size_t budget_size = it->second;
				if (budget_size < aligned_size)	{
					it++;
					continue;
				}

				BufferMemory* ptr = new BufferMemory;
				ptr->buffer = buffer_blocks[i]->buffer;
				ptr->offset = it->first;
				ptr->memory = buffer_blocks[i]->memory;
				ptr->capacity = aligned_size;
				ptr->mapped_ptr = buffer_blocks[i]->mapped_ptr;
				ptr->state = 1;

				if (budget_size == aligned_size)
					budgets[i].erase(it);
				else {
					it->first += aligned_size;
					it->second -= aligned_size;
				}

				return ptr;
			}
		}

		size_t new_block_size = std::max<size_t>(block_size, aligned_size);
		BufferMemory* block = new BufferMemory;
		block->buffer = create_buffer(new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
		block->offset = 0;

		// TODO respect VK_KHR_dedicated_allocation ?

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(dev->vkdevice(), block->buffer, &memoryRequirements);
		block->memory = allocate_memory(memoryRequirements.size, dev->info.device_local_memory_index);
		vkBindBufferMemory(dev->vkdevice(), block->buffer, block->memory, 0);
		block->mapped_ptr = 0;
		if (mappable) {
			vkMapMemory(dev->vkdevice(), block->memory, 0, new_block_size, 0, &block->mapped_ptr);
		}

		buffer_blocks.push_back(block);

		BufferMemory* ptr = new BufferMemory;
		ptr->buffer = block->buffer;
		ptr->offset = 0;
		ptr->memory = block->memory;
		ptr->capacity = aligned_size;
		ptr->mapped_ptr = block->mapped_ptr;
		ptr->state = 1;

		std::list< std::pair<size_t, size_t> > budget;
		if (new_block_size > aligned_size)
			budget.push_back(std::make_pair(aligned_size, new_block_size - aligned_size));
		budgets.push_back(budget);

		return ptr;
	}



	void BlobBufferAllocator::fastFree(BufferMemory* ptr) {
		const int buffer_block_count = buffer_blocks.size();
		int block_index = -1;
		for (int i = 0; i < buffer_block_count; ++i) {
			if (buffer_blocks[i]->buffer == ptr->buffer && buffer_blocks[i]->memory == ptr->memory) {
				block_index = i;
				break;
			}
		}
		if (block_index == -1) {
			fprintf(stderr, "FATAL ERROR! unlocked VkBlobBufferAllocator get wild %p\n", ptr->buffer);
			delete ptr;
			return;
		}
		std::list< std::pair<size_t, size_t> >::iterator it_merge_left = budgets[block_index].end();
		std::list< std::pair<size_t, size_t> >::iterator it_merge_right = budgets[block_index].end();
		std::list< std::pair<size_t, size_t> >::iterator it = budgets[block_index].begin();
		for (; it != budgets[block_index].end(); it++) {
			if (it->first + it->second == ptr->offset)
				it_merge_left = it;
			else if (ptr->offset + ptr->capacity == it->first)
				it_merge_right = it;
		}
		if (it_merge_left != budgets[block_index].end() && it_merge_right != budgets[block_index].end()) {
			it_merge_left->second = it_merge_right->first + it_merge_right->second - it_merge_left->first;
			budgets[block_index].erase(it_merge_right);
		}
		else if (it_merge_left != budgets[block_index].end()) {
			it_merge_left->second = ptr->offset + ptr->capacity - it_merge_left->first;
		}
		else if (it_merge_right != budgets[block_index].end()) {
			it_merge_right->second = it_merge_right->first + it_merge_right->second - ptr->offset;
			it_merge_right->first = ptr->offset;
		}
		else {
			if (ptr->offset == 0) 
				budgets[block_index].push_front(std::make_pair(ptr->offset, ptr->capacity));
			else
				budgets[block_index].push_back(std::make_pair(ptr->offset, ptr->capacity));
		}

		delete ptr;

	}

	WeightBufferAllocator::WeightBufferAllocator(const Device* _dev) : Allocator(_dev) {
		mappable = dev->info.device_local_memory_index == dev->info.unified_memory_index;
		buffer_offset_alignment = dev->info.buffer_offset_alignment;
		if (mappable) {
			size_t memory_map_alignment = dev->info.memory_map_alignment;
			buffer_offset_alignment = least_common_multiple(buffer_offset_alignment, memory_map_alignment);
		}
		block_size = alignSize(8 * 1024 * 1024, buffer_offset_alignment);// 8,388,608‬
	}

	WeightBufferAllocator::~WeightBufferAllocator() { clear(); }

	void WeightBufferAllocator::set_block_size(size_t _block_size) { block_size = _block_size; }

	void WeightBufferAllocator::clear() {
		buffer_block_free_spaces.clear();
		for (size_t i = 0; i < buffer_blocks.size(); ++i) {
			BufferMemory* ptr = buffer_blocks[i];
			if (mappable) vkUnmapMemory(dev->vkdevice(), ptr->memory);
			vkDestroyBuffer(dev->vkdevice(), ptr->buffer, 0);
			vkFreeMemory(dev->vkdevice(), ptr->memory, 0);
			delete ptr;
		}
		buffer_blocks.clear();
		for (size_t i = 0; i < dedicated_buffer_blocks.size(); ++i) {
			BufferMemory* ptr = dedicated_buffer_blocks[i];
			if (mappable) vkUnmapMemory(dev->vkdevice(), ptr->memory);
			vkDestroyBuffer(dev->vkdevice(), ptr->buffer, 0);
			vkFreeMemory(dev->vkdevice(), ptr->memory, 0);
			delete ptr;
		}
		dedicated_buffer_blocks.clear();
	}

	BufferMemory* WeightBufferAllocator::fastMalloc(size_t size) {
		size_t aligned_size = alignSize(size, buffer_offset_alignment);
		const int buffer_block_count = buffer_blocks.size();
		int block_index = -1;
		size_t block_offset = 0;
		for (int i = 0; i < buffer_block_count; ++i) {
			size_t free_size = buffer_block_free_spaces[i];
			if (free_size >= aligned_size) {
				block_index = i;
				block_offset = block_size - free_size;
				break;
			}
		}

		if (block_index != -1) {
			BufferMemory* ptr = new BufferMemory;
			ptr->buffer = buffer_blocks[block_index]->buffer;
			ptr->offset = block_offset;
			ptr->memory = buffer_blocks[block_index]->memory;
			ptr->capacity = aligned_size;
			ptr->mapped_ptr = buffer_blocks[block_index]->mapped_ptr;
			ptr->state = 1;
			buffer_block_free_spaces[block_index] -= aligned_size;
			return ptr;
		}

		size_t new_block_size = std::max<size_t>(block_size, aligned_size);
		BufferMemory* block = new BufferMemory;
		block->buffer = create_buffer(new_block_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
		block->offset = 0;

		if (dev->info.support_VK_KHR_get_memory_requirements2 && dev->info.support_VK_KHR_dedicated_allocation) {
			VkBufferMemoryRequirementsInfo2KHR bufferMemoryRequirementsInfo2;
			bufferMemoryRequirementsInfo2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2_KHR;
			bufferMemoryRequirementsInfo2.pNext = 0;
			bufferMemoryRequirementsInfo2.buffer = block->buffer;

			VkMemoryRequirements2KHR memoryRequirements2;
			memoryRequirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2_KHR;
			memoryRequirements2.pNext = 0;

			VkMemoryDedicatedRequirementsKHR memoryDedicatedRequirements;
			memoryDedicatedRequirements.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS_KHR;
			memoryDedicatedRequirements.pNext = 0;
			memoryRequirements2.pNext = &memoryDedicatedRequirements;

			dev->vkGetBufferMemoryRequirements2KHR(dev->vkdevice(), &bufferMemoryRequirementsInfo2, &memoryRequirements2);
			bool dedicatedAllocation = memoryDedicatedRequirements.requiresDedicatedAllocation || memoryDedicatedRequirements.prefersDedicatedAllocation;
			if (dedicatedAllocation){
				block->memory = allocate_dedicated_memory(memoryRequirements2.memoryRequirements.size, dev->info.device_local_memory_index, block->buffer);
				vkBindBufferMemory(dev->vkdevice(), block->buffer, block->memory, 0);
				block->mapped_ptr = 0;
				if (mappable)
					vkMapMemory(dev->vkdevice(), block->memory, 0, new_block_size, 0, &block->mapped_ptr);
				dedicated_buffer_blocks.push_back(block);
				BufferMemory* ptr = new BufferMemory;
				ptr->buffer = block->buffer;
				ptr->offset = 0;
				ptr->memory = block->memory;
				ptr->capacity = new_block_size;
				ptr->mapped_ptr = block->mapped_ptr;
				ptr->state = 1;
				return ptr;
			}

			VkMemoryRequirements memoryRequirements;
			vkGetBufferMemoryRequirements(dev->vkdevice(), block->buffer, &memoryRequirements);
			block->memory = allocate_memory(memoryRequirements.size, dev->info.device_local_memory_index);
			vkBindBufferMemory(dev->vkdevice(), block->buffer, block->memory, 0);
			block->mapped_ptr = 0;
			if (mappable)
				vkMapMemory(dev->vkdevice(), block->memory, 0, new_block_size, 0, &block->mapped_ptr);
			buffer_blocks.push_back(block);
			buffer_block_free_spaces.push_back(new_block_size - aligned_size);
			BufferMemory* ptr = new BufferMemory;
			ptr->buffer = block->buffer;
			ptr->offset = 0;
			ptr->memory = block->memory;
			ptr->capacity = aligned_size;
			ptr->mapped_ptr = block->mapped_ptr;
			ptr->state = 1;
			return ptr;
		}
	}

	void WeightBufferAllocator::fastFree(BufferMemory* ptr) {
		delete ptr;
	}

	StagingBufferAllocator::StagingBufferAllocator(const Device* _dev) : Allocator(_dev) {
		mappable = true;
		memory_type_index = dev->info.unified_memory_index;
		if (memory_type_index == -1)
			memory_type_index = dev->info.host_visible_memory_index;
		size_compare_ratio = 192;// 0.75f * 256
	}

	StagingBufferAllocator::~StagingBufferAllocator() { clear(); }

	void StagingBufferAllocator::set_size_compare_ratio(float scr) {
		if (scr < 0.f || scr>1.0f)
			return;
		size_compare_ratio = (unsigned int)(scr * 256);
	}

	void StagingBufferAllocator::clear() {
		std::list<BufferMemory*>::iterator it = budgets.begin();
		for (; it != budgets.end(); ++it) {
			BufferMemory* ptr = *it;
			vkUnmapMemory(dev->vkdevice(), ptr->memory);
			vkDestroyBuffer(dev->vkdevice(), ptr->buffer, 0);
			vkFreeMemory(dev->vkdevice(), ptr->memory, 0);
			delete ptr;
		}
		budgets.clear();
	}

	BufferMemory* StagingBufferAllocator::fastMalloc(size_t size) {
		std::list<BufferMemory*>::iterator it = budgets.begin();
		for (; it != budgets.end(); it++) {
			BufferMemory* ptr = *it;
			size_t capacity = ptr->capacity;
			if (capacity >= size && ((capacity * size_compare_ratio) >> 8) <= size) {
				budgets.erase(it);
				return ptr;
			}
		}
		BufferMemory* ptr = new BufferMemory;
		ptr->buffer = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT || VK_BUFFER_USAGE_TRANSFER_DST_BIT);
		ptr->offset = 0;
		

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(dev->vkdevice(), ptr->buffer, &memoryRequirements);
		ptr->memory = allocate_memory(memoryRequirements.size, memory_type_index);
		vkBindBufferMemory(dev->vkdevice(), ptr->buffer, ptr->memory, 0);
		ptr->capacity = size;
		vkMapMemory(dev->vkdevice(), ptr->memory, 0, size, 0, &ptr->mapped_ptr);
		ptr->state = 1;

		return ptr;
	}

	void StagingBufferAllocator::fastFree(BufferMemory* ptr) {
		budgets.push_back(ptr);
	}

	WeightStagingBufferAllocator::WeightStagingBufferAllocator(const Device* _dev) : Allocator(_dev) {
		mappable = true;
		memory_type_index = dev->info.host_visible_memory_index;
	}

	WeightStagingBufferAllocator::~WeightStagingBufferAllocator() {}

	BufferMemory* WeightStagingBufferAllocator::fastMalloc(size_t size) {
		BufferMemory* ptr = new BufferMemory;
		ptr->buffer = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
		ptr->offset = 0;
		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(dev->vkdevice(), ptr->buffer, &memoryRequirements);
		ptr->memory = allocate_memory(memoryRequirements.size, memory_type_index);
		vkBindBufferMemory(dev->vkdevice(), ptr->buffer, ptr->memory, 0);
		ptr->capacity = size;
		vkMapMemory(dev->vkdevice(), ptr->memory, 0, size, 0, &ptr->mapped_ptr);
		ptr->state = 1;
		return ptr;
	}

	void WeightStagingBufferAllocator::fastFree(BufferMemory* ptr) {
		vkUnmapMemory(dev->vkdevice(), ptr->memory);
		vkDestroyBuffer(dev->vkdevice(), ptr->buffer, 0);
		vkFreeMemory(dev->vkdevice(), ptr->memory, 0);
		delete ptr;
	}

}
