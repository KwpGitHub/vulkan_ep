#include "device.h"
#include "allocator.h"

#define ENABLE_VALIDATION_LAYER 0

namespace backend {
	static VkInstance instance = 0;
	static int device_count = 0;
	static int default_gpu_index = -1;

#define VK_EP_MAX_GPU_COUNT 8
	static DeviceInfo device_infos[VK_EP_MAX_GPU_COUNT];

	static Mutex default_dev_lock;
	static Device* default_dev[VK_EP_MAX_GPU_COUNT] = { 0 };

	int support_VK_KHR_get_physical_device_properties2 = 0;
	int support_VK_EXT_debug_utils = 0;

	PFN_vkGetPhysicalDeviceFeatures2KHR vkGetPhysicalDeviceFeatures2KHR = 0;
	PFN_vkGetPhysicalDeviceProperties2KHR vkGetPhysicalDeviceProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceFormatProperties2KHR vkGetPhysicalDeviceFormatProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceImageFormatProperties2KHR vkGetPhysicalDeviceImageFormatProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR vkGetPhysicalDeviceQueueFamilyProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceMemoryProperties2KHR vkGetPhysicalDeviceMemoryProperties2KHR = 0;
	PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR vkGetPhysicalDeviceSparseImageFormatProperties2KHR = 0;

	static int init_instance_extension() {
		if (support_VK_KHR_get_physical_device_properties2) {
			vkGetPhysicalDeviceFeatures2KHR = (PFN_vkGetPhysicalDeviceFeatures2KHR)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2KHR");
			vkGetPhysicalDeviceProperties2KHR = (PFN_vkGetPhysicalDeviceProperties2KHR)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2KHR");
			vkGetPhysicalDeviceFormatProperties2KHR = (PFN_vkGetPhysicalDeviceFormatProperties2KHR)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFormatProperties2KHR");
			vkGetPhysicalDeviceImageFormatProperties2KHR = (PFN_vkGetPhysicalDeviceImageFormatProperties2KHR)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceImageFormatProperties2KHR");
			vkGetPhysicalDeviceQueueFamilyProperties2KHR = (PFN_vkGetPhysicalDeviceQueueFamilyProperties2KHR)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceQueueFamilyProperties2KHR");
			vkGetPhysicalDeviceMemoryProperties2KHR = (PFN_vkGetPhysicalDeviceMemoryProperties2KHR)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceMemoryProperties2KHR");
			vkGetPhysicalDeviceSparseImageFormatProperties2KHR = (PFN_vkGetPhysicalDeviceSparseImageFormatProperties2KHR)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceSparseImageFormatProperties2KHR");
		}

		return 0;
	}

#if ENABLE_VALIDATION_LAYER
	static VkDebugUtilsMessengerEXT callback;

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
		VkDebugUtilsMessageSeverityFlagBitsEXT /*messageSeverity*/,
		VkDebugUtilsMessageTypeFlagsEXT /*messageType*/,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* /*pUserData*/)
	{
		fprintf(stderr, "validation layer: %s\n", pCallbackData->pMessage);

		return VK_FALSE;
	}

	VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pCallback) {
		PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
		if (func) return func(instance, pCreateInfo, pAllocator, pCallback);
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}

	void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* pAllocator) {
		PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
		if (func) func(instance, callback, pAllocator);
	}
#endif  // ENABLE_VALIDATION_LAYER

	static uint32_t find_device_compute_queue(const std::vector<VkQueueFamilyProperties>& queueFamilyProperties) {
		for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];
			if ((queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT) && !(queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
				return i;
			}
		}

		for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];
			if (queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT) {
				return i;
			}
		}

		return -1;
	}


	static uint32_t find_device_transfer_queue(const std::vector<VkQueueFamilyProperties>& queueFamilyProperties)
	{
		for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];
			if ((queueFamilyProperty.queueFlags & VK_QUEUE_TRANSFER_BIT) && !(queueFamilyProperty.queueFlags & VK_QUEUE_COMPUTE_BIT) && !(queueFamilyProperty.queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
				return i;
			}
		}

		for (uint32_t i = 0; i < queueFamilyProperties.size(); ++i) {
			const VkQueueFamilyProperties& queueFamilyProperty = queueFamilyProperties[i];
			if (queueFamilyProperty.queueFlags & VK_QUEUE_TRANSFER_BIT) {
				return i;
			}
		}

		uint32_t compute_queue_index = find_device_compute_queue(queueFamilyProperties);
		if (compute_queue_index != (uint32_t)-1) {
			return compute_queue_index;
		}

		return -1;
	}

	static uint32_t find_unified_memory(VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties)
	{
		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; ++i) {
			const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];
			if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) && (memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
				return i;
			}
		}

		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; ++i)
		{
			const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];

			if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
				return i;
			}
		}

		return -1;
	}

	static uint32_t find_device_local_memory(VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties)
	{
		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; ++i) {
			const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];
			if (memoryType.propertyFlags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
				return i;
			}
		}

		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; ++i) {
			const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];
			if (memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
				return i;
			}
		}

		return -1;
	}

	static uint32_t find_host_visible_memory(VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties)
	{
		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; ++i) {
			const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];
			if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) && !(memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
				return i;
			}
		}

		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; ++i) {
			const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];
			if ((memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && !(memoryType.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
				return i;
			}
		}

		for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; ++i) {
			const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[i];
			if (memoryType.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
				return i;
			}
		}

		return -1;
	}

	static int find_default_vulkan_device_index()
	{
		for (int i = 0; i < device_count; ++i) {
			if (device_infos[i].type == 0)
				return i;
		}

		for (int i = 0; i < device_count; ++i) {
			if (device_infos[i].type == 1)
				return i;
		}

		if (device_infos > 0)
			return 0;

		return -1;
	}

	int create_gpu_instance() {
		VkResult ret;
		std::vector<const char*> enabledLayers;
		std::vector<const char*> enabledExtensions;

#if ENABLE_VALIDATION_LAYER
		uint32_t instanceLayerPropertyCount;
		ret = vkEnumerateInstanceLayerProperties(&instanceLayerPropertyCount, NULL);
		if (ret != VK_SUCCESS)
			return -1;

		std::vector<VkLayerProperties> instanceLayerProperties(instanceLayerPropertyCount);
		ret = vkEnumerateInstanceLayerProperties(&instanceLayerPropertyCount, instanceLayerProperties.data());
		if (ret != VK_SUCCESS) return -1;

		for (uint32_t i = 0; i < instanceLayerPropertyCount; ++i)
		{
			const VkLayerProperties& lp = instanceLayerProperties[i];
			if (strcmp(lp.layerName, "VK_LAYER_LUNARG_standard_validation") == 0)
				enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");

			if (strcmp(lp.layerName, "VK_LAYER_LUNARG_parameter_validation") == 0)
				enabledLayers.push_back("VK_LAYER_LUNARG_parameter_validation");
		}
#endif // ENABLE_VALIDATION_LAYER

		uint32_t instanceExtensionPropertyCount;
		ret = vkEnumerateInstanceExtensionProperties(NULL, &instanceExtensionPropertyCount, NULL);
		if (ret != VK_SUCCESS) return -1;

		std::vector<VkExtensionProperties> instanceExtensionProperties(instanceExtensionPropertyCount);
		ret = vkEnumerateInstanceExtensionProperties(NULL, &instanceExtensionPropertyCount, instanceExtensionProperties.data());
		if (ret != VK_SUCCESS) return -1;

		support_VK_KHR_get_physical_device_properties2 = 0;
		support_VK_EXT_debug_utils = 0;
		for (uint32_t j = 0; j < instanceExtensionPropertyCount; ++j) {
			const VkExtensionProperties& exp = instanceExtensionProperties[j];
			if (strcmp(exp.extensionName, "VK_KHR_get_physical_device_properties2") == 0)
				support_VK_KHR_get_physical_device_properties2 = exp.specVersion;
			if (strcmp(exp.extensionName, "VK_EXT_debug_utils") == 0)
				support_VK_EXT_debug_utils = exp.specVersion;
		}

		if (support_VK_KHR_get_physical_device_properties2)
			enabledExtensions.push_back("VK_KHR_get_physical_device_properties2");
#if ENABLE_VALIDATION_LAYER
		if (support_VK_EXT_debug_utils)
			enabledExtensions.push_back("VK_EXT_debug_utils");
#endif // ENABLE_VALIDATION_LAYER

		VkApplicationInfo applicationInfo;
		applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		applicationInfo.pNext = 0;
		applicationInfo.pApplicationName = "vulkan_ep";
		applicationInfo.applicationVersion = 0;
		applicationInfo.pEngineName = "vulkan_ep";
		applicationInfo.engineVersion = 20190319;
		applicationInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

		VkInstanceCreateInfo instanceCreateInfo;
		instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instanceCreateInfo.pNext = 0;
		instanceCreateInfo.flags = 0;
		instanceCreateInfo.pApplicationInfo = &applicationInfo;
		instanceCreateInfo.enabledLayerCount = enabledLayers.size();
		instanceCreateInfo.ppEnabledLayerNames = enabledLayers.data();
		instanceCreateInfo.enabledExtensionCount = enabledExtensions.size();
		instanceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();

		ret = vkCreateInstance(&instanceCreateInfo, 0, &instance);
		if (ret != VK_SUCCESS) return -1;


#if ENABLE_VALIDATION_LAYER
		if (support_VK_EXT_debug_utils)
		{
			VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			createInfo.pfnUserCallback = debugCallback;
			createInfo.pUserData = 0;
			ret = CreateDebugUtilsMessengerEXT(instance, &createInfo, NULL, &callback);
			if (ret != VK_SUCCESS)
			{
				fprintf(stderr, "CreateDebugUtilsMessengerEXT failed %d\n", ret);
				return -1;
			}
		}
#endif // ENABLE_VALIDATION_LAYER

		init_instance_extension();
		uint32_t physicalDeviceCount = 0;
		ret = vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, 0);
		if (ret != VK_SUCCESS) return -1;

		if (physicalDeviceCount > VK_EP_MAX_GPU_COUNT)
			physicalDeviceCount = VK_EP_MAX_GPU_COUNT;

		std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
		ret = vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, physicalDevices.data());
		if (ret != VK_SUCCESS) return -1;

		if (ret != VK_SUCCESS) return -1;
		int gpu_info_index = 0;

		for (uint32_t i = 0; i < physicalDeviceCount; ++i)
		{
			const VkPhysicalDevice& physicalDevice = physicalDevices[i];
			DeviceInfo& gpu_info = device_infos[gpu_info_index];

			// device type
			VkPhysicalDeviceProperties physicalDeviceProperties;
			vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

			gpu_info.physical_device = physicalDevice;
			gpu_info.api_version = physicalDeviceProperties.apiVersion;
			gpu_info.driver_version = physicalDeviceProperties.driverVersion;
			gpu_info.vendor_id = physicalDeviceProperties.vendorID;
			gpu_info.device_id = physicalDeviceProperties.deviceID;
			memcpy(gpu_info.pipeline_cache_uuid, physicalDeviceProperties.pipelineCacheUUID, VK_UUID_SIZE);

			if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
				gpu_info.type = 0;
			else if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
				gpu_info.type = 1;
			else if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
				gpu_info.type = 2;
			else if (physicalDeviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
				gpu_info.type = 3;
			else
				gpu_info.type = -1;

			gpu_info.max_shared_memory_size = physicalDeviceProperties.limits.maxComputeSharedMemorySize;
			gpu_info.max_workgroup_count[0] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[0];
			gpu_info.max_workgroup_count[1] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[1];
			gpu_info.max_workgroup_count[2] = physicalDeviceProperties.limits.maxComputeWorkGroupCount[2];
			gpu_info.max_workgroup_invocations = physicalDeviceProperties.limits.maxComputeWorkGroupInvocations;
			gpu_info.max_workgroup_size[0] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[0];
			gpu_info.max_workgroup_size[1] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[1];
			gpu_info.max_workgroup_size[2] = physicalDeviceProperties.limits.maxComputeWorkGroupSize[2];
			gpu_info.memory_map_alignment = physicalDeviceProperties.limits.minMemoryMapAlignment;
			gpu_info.buffer_offset_alignment = physicalDeviceProperties.limits.minStorageBufferOffsetAlignment;
			gpu_info.timestamp_period = physicalDeviceProperties.limits.timestampPeriod;

			uint32_t queueFamilyPropertiesCount;
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, 0);
			std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyPropertiesCount);
			vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyPropertiesCount, queueFamilyProperties.data());

			gpu_info.compute_queue_family_index = find_device_compute_queue(queueFamilyProperties);
			gpu_info.transfer_queue_family_index = find_device_transfer_queue(queueFamilyProperties);
			gpu_info.compute_queue_count = queueFamilyProperties[gpu_info.compute_queue_family_index].queueCount;
			gpu_info.transfer_queue_count = queueFamilyProperties[gpu_info.transfer_queue_family_index].queueCount;

			VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
			vkGetPhysicalDeviceMemoryProperties(physicalDevice, &physicalDeviceMemoryProperties);


			//         // print memory info
			//         for (uint32_t j=0; j<physicalDeviceMemoryProperties.memoryTypeCount; j++)
			//         {
			//             const VkMemoryType& memoryType = physicalDeviceMemoryProperties.memoryTypes[j];
			//             fprintf(stderr, "[%u] memoryType %u heapIndex/propertyFlags = %d  %u\n", i, j, memoryType.heapIndex, memoryType.propertyFlags);
			//         }
			//         for (uint32_t j=0; j<physicalDeviceMemoryProperties.memoryHeapCount; j++)
			//         {
			//             const VkMemoryHeap& memoryHeap = physicalDeviceMemoryProperties.memoryHeaps[j];
			//             fprintf(stderr, "[%u] memoryHeap %u size/flags = %lu  %u\n", i, j, memoryHeap.size, memoryHeap.flags);
			//         }

			gpu_info.unified_memory_index = find_unified_memory(physicalDeviceMemoryProperties);
			gpu_info.device_local_memory_index = find_device_local_memory(physicalDeviceMemoryProperties);
			gpu_info.host_visible_memory_index = find_host_visible_memory(physicalDeviceMemoryProperties);

			if (gpu_info.unified_memory_index != (uint32_t)-1) {
				int unified_memory_heap_index = physicalDeviceMemoryProperties.memoryTypes[gpu_info.unified_memory_index].heapIndex;
				int device_local_memory_heap_index = physicalDeviceMemoryProperties.memoryTypes[gpu_info.device_local_memory_index].heapIndex;
				int host_visible_memory_heap_index = physicalDeviceMemoryProperties.memoryTypes[gpu_info.host_visible_memory_index].heapIndex;
				if (unified_memory_heap_index == device_local_memory_heap_index && unified_memory_heap_index == host_visible_memory_heap_index) {
					gpu_info.device_local_memory_index = gpu_info.unified_memory_index;
					gpu_info.host_visible_memory_index = gpu_info.unified_memory_index;
				}
			}

			uint32_t deviceExtensionPropertyCount = 0;
			ret = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &deviceExtensionPropertyCount, NULL);
			if (ret != VK_SUCCESS) return -1;
			std::vector<VkExtensionProperties> deviceExtensionProperties(deviceExtensionPropertyCount);
			ret = vkEnumerateDeviceExtensionProperties(physicalDevice, NULL, &deviceExtensionPropertyCount, deviceExtensionProperties.data());
			if (ret != VK_SUCCESS) return -1;

			gpu_info.support_VK_KHR_8bit_storage = 0;
			gpu_info.support_VK_KHR_16bit_storage = 0;
			gpu_info.support_VK_KHR_bind_memory2 = 0;
			gpu_info.support_VK_KHR_dedicated_allocation = 0;
			gpu_info.support_VK_KHR_descriptor_update_template = 0;
			gpu_info.support_VK_KHR_get_memory_requirements2 = 0;
			gpu_info.support_VK_KHR_push_descriptor = 0;
			gpu_info.support_VK_KHR_shader_float16_int8 = 0;
			gpu_info.support_VK_KHR_shader_float_controls = 0;
			gpu_info.support_VK_KHR_storage_buffer_storage_class = 0;

			for (uint32_t j = 0; j < deviceExtensionPropertyCount; ++j) {
				const VkExtensionProperties& exp = deviceExtensionProperties[j];
				if (strcmp(exp.extensionName, "VK_KHR_8bit_storage") == 0)
					gpu_info.support_VK_KHR_8bit_storage = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_16bit_storage") == 0)
					gpu_info.support_VK_KHR_16bit_storage = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_bind_memory2") == 0)
					gpu_info.support_VK_KHR_bind_memory2 = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_dedicated_allocation") == 0)
					gpu_info.support_VK_KHR_dedicated_allocation = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_descriptor_update_template") == 0)
					gpu_info.support_VK_KHR_descriptor_update_template = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_get_memory_requirements2") == 0)
					gpu_info.support_VK_KHR_get_memory_requirements2 = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_push_descriptor") == 0)
					gpu_info.support_VK_KHR_push_descriptor = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_shader_float16_int8") == 0)
					gpu_info.support_VK_KHR_shader_float16_int8 = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_shader_float_controls") == 0)
					gpu_info.support_VK_KHR_shader_float_controls = exp.specVersion;
				else if (strcmp(exp.extensionName, "VK_KHR_storage_buffer_storage_class") == 0)
					gpu_info.support_VK_KHR_storage_buffer_storage_class = exp.specVersion;
			}

			gpu_info.support_fp16_packed = true;
			gpu_info.support_fp16_storage = false;
			gpu_info.support_fp16_arithmetic = false;
			gpu_info.support_int8_storage = false;
			gpu_info.support_int8_arithmetic = false;

			if (support_VK_KHR_get_physical_device_properties2) {
				void* queryExtensionFeatures = 0;

				VkPhysicalDevice8BitStorageFeaturesKHR query8BitStorageFeatures;
				query8BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR;
				query8BitStorageFeatures.pNext = 0;
				if (gpu_info.support_VK_KHR_8bit_storage) {
					query8BitStorageFeatures.pNext = queryExtensionFeatures;
					queryExtensionFeatures = &query8BitStorageFeatures;
				}
				VkPhysicalDevice16BitStorageFeaturesKHR query16BitStorageFeatures;
				query16BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR;
				query16BitStorageFeatures.pNext = 0;
				if (gpu_info.support_VK_KHR_16bit_storage) {
					query16BitStorageFeatures.pNext = queryExtensionFeatures;
					queryExtensionFeatures = &query16BitStorageFeatures;
				}
				VkPhysicalDeviceFloat16Int8FeaturesKHR queryFloat16Int8Features;
				queryFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
				queryFloat16Int8Features.pNext = 0;
				if (gpu_info.support_VK_KHR_shader_float16_int8) {
					queryFloat16Int8Features.pNext = queryExtensionFeatures;
					queryExtensionFeatures = &queryFloat16Int8Features;
				}
				VkPhysicalDeviceFeatures2KHR queryFeatures;
				queryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR,
					queryFeatures.pNext = queryExtensionFeatures;

				vkGetPhysicalDeviceFeatures2KHR(physicalDevice, &queryFeatures);

				if (gpu_info.support_VK_KHR_8bit_storage)
					gpu_info.support_int8_storage = query8BitStorageFeatures.storageBuffer8BitAccess && query8BitStorageFeatures.uniformAndStorageBuffer8BitAccess;
				if (gpu_info.support_VK_KHR_16bit_storage)
					gpu_info.support_fp16_storage = query16BitStorageFeatures.storageBuffer16BitAccess && query16BitStorageFeatures.uniformAndStorageBuffer16BitAccess;
				if (gpu_info.support_VK_KHR_shader_float16_int8) {
					gpu_info.support_fp16_arithmetic = queryFloat16Int8Features.shaderFloat16;
					gpu_info.support_int8_arithmetic = queryFloat16Int8Features.shaderInt8;
				}
			}
			else {
				//             // TODO
				//             VkPhysicalDeviceFeatures features;
				//             vkGetPhysicalDeviceFeatures(physicalDevice, &features);
			}

			if (physicalDeviceProperties.vendorID == 0x13b5) {
				gpu_info.support_fp16_storage = false;
			}


			gpu_info_index++;

		}

		device_count = gpu_info_index;
		default_gpu_index = find_default_vulkan_device_index();
		return 0;
	}


	void destroy_gpu_instance() {
		for (int i = 0; i < VK_EP_MAX_GPU_COUNT; ++i) {
			delete default_dev[i];
			default_dev[i] = 0;
		}

#if ENABLE_VALIDATION_LAYER
		if (support_VK_EXT_debug_utils)
		{
			DestroyDebugUtilsMessengerEXT(g_instance, callback, NULL);
		}
#endif // ENABLE_VALIDATION_LAYER

		vkDestroyInstance(instance, 0);
	}

	int get_gpu_count() {
		return device_count;
	}

	int get_default_gpu_index() {
		return default_gpu_index;
	}

	const DeviceInfo& get_gpu_info(int device_index) {
		return device_infos[device_index];
	}

	struct layer_shader_registry_entry {
		const char* name;
		const uint32_t* spv_data;
		size_t spv_data_size;
	};

#include "layer_shader_spv_data.h"
	static const layer_shader_registry_entry layer_shader_registry[] = {
#include "layer_shader_registry.h"
	};

	static const int layer_shader_registry_entry_count = sizeof(layer_shader_registry) / sizeof(layer_shader_registry_entry);

}


namespace backend {
	Device::Device(int device_index) :info(device_infos[device_index]) {
		std::vector<const char*> enabledExtensions;
		if (info.support_VK_KHR_8bit_storage)
			enabledExtensions.push_back("VK_KHR_8bit_storage");
		if (info.support_VK_KHR_16bit_storage)
			enabledExtensions.push_back("VK_KHR_16bit_storage");
		if (info.support_VK_KHR_bind_memory2)
			enabledExtensions.push_back("VK_KHR_bind_memory2");
		if (info.support_VK_KHR_dedicated_allocation)
			enabledExtensions.push_back("VK_KHR_dedicated_allocation");
		if (info.support_VK_KHR_descriptor_update_template)
			enabledExtensions.push_back("VK_KHR_descriptor_update_template");
		if (info.support_VK_KHR_get_memory_requirements2)
			enabledExtensions.push_back("VK_KHR_get_memory_requirements2");
		if (info.support_VK_KHR_push_descriptor)
			enabledExtensions.push_back("VK_KHR_push_descriptor");
		if (info.support_VK_KHR_shader_float16_int8)
			enabledExtensions.push_back("VK_KHR_shader_float16_int8");
		if (info.support_VK_KHR_shader_float_controls)
			enabledExtensions.push_back("VK_KHR_shader_float_controls");
		if (info.support_VK_KHR_storage_buffer_storage_class)
			enabledExtensions.push_back("VK_KHR_storage_buffer_storage_class");

		void* enabledExtensionFeatures = 0;

		VkPhysicalDevice8BitStorageFeaturesKHR enabled8BitStorageFeatures;
		enabled8BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES_KHR;
		enabled8BitStorageFeatures.pNext = 0;
		enabled8BitStorageFeatures.storageBuffer8BitAccess = info.support_int8_storage;
		enabled8BitStorageFeatures.uniformAndStorageBuffer8BitAccess = info.support_int8_storage;
		enabled8BitStorageFeatures.storagePushConstant8 = VK_FALSE;
		if (support_VK_KHR_get_physical_device_properties2 && info.support_VK_KHR_8bit_storage) {
			enabled8BitStorageFeatures.pNext = enabledExtensionFeatures;
			enabledExtensionFeatures = &enabled8BitStorageFeatures;
		}

		VkPhysicalDevice16BitStorageFeaturesKHR enabled16BitStorageFeatures;
		enabled16BitStorageFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR;
		enabled16BitStorageFeatures.pNext = 0;
		enabled16BitStorageFeatures.storageBuffer16BitAccess = info.support_fp16_storage;
		enabled16BitStorageFeatures.uniformAndStorageBuffer16BitAccess = info.support_fp16_storage;
		enabled16BitStorageFeatures.storagePushConstant16 = VK_FALSE;
		enabled16BitStorageFeatures.storageInputOutput16 = VK_FALSE;
		if (support_VK_KHR_get_physical_device_properties2 && info.support_VK_KHR_16bit_storage) {
			enabled16BitStorageFeatures.pNext = enabledExtensionFeatures;
			enabledExtensionFeatures = &enabled16BitStorageFeatures;
		}

		VkPhysicalDeviceFloat16Int8FeaturesKHR enabledFloat16Int8Features;
		enabledFloat16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
		enabledFloat16Int8Features.pNext = 0;
		enabledFloat16Int8Features.shaderFloat16 = info.support_fp16_arithmetic;
		enabledFloat16Int8Features.shaderInt8 = info.support_int8_arithmetic;
		if (support_VK_KHR_get_physical_device_properties2 && info.support_VK_KHR_shader_float16_int8) {
			enabledFloat16Int8Features.pNext = enabledExtensionFeatures;
			enabledExtensionFeatures = &enabledFloat16Int8Features;
		}

		std::vector<float> compute_queue_priorities(info.compute_queue_count, 1.f);// 0.f ~ 1.f
		std::vector<float> transfer_queue_priorities(info.transfer_queue_count, 1.f);// 0.f ~ 1.f

		VkDeviceQueueCreateInfo deviceQueueCreateInfos[2];
		deviceQueueCreateInfos[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		deviceQueueCreateInfos[0].pNext = 0;
		deviceQueueCreateInfos[0].flags = 0;
		deviceQueueCreateInfos[0].queueFamilyIndex = info.compute_queue_family_index;
		deviceQueueCreateInfos[0].queueCount = info.compute_queue_count;
		deviceQueueCreateInfos[0].pQueuePriorities = compute_queue_priorities.data();
		deviceQueueCreateInfos[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		deviceQueueCreateInfos[1].pNext = 0;
		deviceQueueCreateInfos[1].flags = 0;
		deviceQueueCreateInfos[1].queueFamilyIndex = info.transfer_queue_family_index;
		deviceQueueCreateInfos[1].queueCount = info.transfer_queue_count;
		deviceQueueCreateInfos[1].pQueuePriorities = transfer_queue_priorities.data();

		VkDeviceCreateInfo deviceCreateInfo;
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.pNext = enabledExtensionFeatures;
		deviceCreateInfo.flags = 0;

		if (info.compute_queue_family_index == info.transfer_queue_family_index)
			deviceCreateInfo.queueCreateInfoCount = 1;
		else
			deviceCreateInfo.queueCreateInfoCount = 2;

		deviceCreateInfo.pQueueCreateInfos = deviceQueueCreateInfos;
		deviceCreateInfo.enabledLayerCount = 0;
		deviceCreateInfo.ppEnabledLayerNames = 0;
		deviceCreateInfo.enabledExtensionCount = enabledExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = enabledExtensions.data();
		deviceCreateInfo.pEnabledFeatures = 0; // VkPhysicalDeviceFeatures pointer

		VkResult ret = vkCreateDevice(info.physical_device, &deviceCreateInfo, 0, &device);
		if (ret != VK_SUCCESS) return;

		init_device_extension();
		create_shader_module();

		compute_queues.resize(info.compute_queue_count);
		blob_allocators.resize(info.compute_queue_count);
		staging_allocators.resize(info.compute_queue_count);
		for (uint32_t i = 0; i < info.compute_queue_count; ++i) {
			vkGetDeviceQueue(device, info.compute_queue_family_index, i, &compute_queues[i]);
			blob_allocators[i] = new BlobBufferAllocator(this);
			staging_allocators[i] = new StagingBufferAllocator(this);
		}

		if (info.compute_queue_family_index != info.transfer_queue_family_index) {
			transfer_queues.resize(info.transfer_queue_count);
			for (uint32_t i = 0; i < info.transfer_queue_count; ++i) {
				vkGetDeviceQueue(device, info.transfer_queue_family_index, i, &transfer_queues[i]);
			}
		}
	}

	Device::~Device() {
		for (uint32_t i = 0; i < info.compute_queue_count; ++i) {
			delete blob_allocators[i];
			delete staging_allocators[i];
		}
		blob_allocators.clear();
		staging_allocators.clear();
		destroy_shader_module();
		vkDestroyDevice(device, 0);
	}

	VkShaderModule Device::get_shader_module(const char* name) const {
		for (int i = 0; i < layer_shader_registry_entry_count; ++i) {
			if (strcmp(layer_shader_registry[i].name, name) == 0)
				return shader_modules[i];
		}

		return 0;
	}

	VkShaderModule Device::compile_shader_module(const uint32_t* spv_data, size_t spv_data_size) const {
		VkShaderModuleCreateInfo shaderModuleCreateInfo;
		shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		shaderModuleCreateInfo.pNext = 0;
		shaderModuleCreateInfo.flags = 0;
		shaderModuleCreateInfo.codeSize = spv_data_size;
		shaderModuleCreateInfo.pCode = spv_data;
		
		VkShaderModule shader_module;
		VkResult ret = vkCreateShaderModule(device, &shaderModuleCreateInfo, 0, &shader_module);
		if (ret != VK_SUCCESS) return 0;
		return shader_module;
	}

	VkQueue Device::acquire_queue(uint32_t queue_family_index) const {
		if (queue_family_index != info.compute_queue_family_index && queue_family_index != info.transfer_queue_family_index)
			return 0;

		MutexLockGuard lock(queue_lock);

		std::vector<VkQueue>& queues = queue_family_index == info.compute_queue_family_index ? compute_queues : transfer_queues;
		for (int i = 0; i < (int)queues.size(); ++i) {
			VkQueue queue = queues[i];
			if (queue) {
				queues[i] = 0;
				return queue;
			}
		}

		return 0;
	}

	void Device::reclaim_queue(uint32_t queue_family_index, VkQueue queue) const {
		if (queue_family_index != info.compute_queue_family_index && queue_family_index != info.transfer_queue_family_index) {
			return;
		}

		MutexLockGuard lock(queue_lock);

		std::vector<VkQueue>& queues = queue_family_index == info.compute_queue_family_index ? compute_queues : transfer_queues;
		for (int i = 0; i < (int)queues.size(); ++i) {
			if (!queues[i]) {
				queues[i] = queue;
				return;
			}
		}
	}

	Allocator* Device::acquire_blob_allocator() const {
		MutexLockGuard lock(blob_allocator_lock);
		for (int i = 0; i < (int)blob_allocators.size(); ++i) {
			Allocator* allocator = blob_allocators[i];
			if (allocator) {
				blob_allocators[i] = 0;
				return allocator;
			}
		}
		return 0;
	}


	void Device::reclaim_blob_allocator(Allocator* allocator) const {
		MutexLockGuard lock(blob_allocator_lock);
		for (int i = 0; i < (int)blob_allocators.size(); ++i) {
			if (!blob_allocators[i]) {
				blob_allocators[i] = allocator;
				return;
			}
		}
	}

	Allocator* Device::acquire_staging_allocator() const {
		MutexLockGuard lock(staging_allocator_lock);
		for (int i = 0; i < (int)staging_allocators.size(); ++i) {
			Allocator* allocator = staging_allocators[i];
			if (allocator){
				staging_allocators[i] = 0;
				return allocator;
			}
		}

		return 0;
	}

	void Device::reclaim_staging_allocator(Allocator* allocator) const {
		MutexLockGuard lock(staging_allocator_lock);
		for (int i = 0; i < (int)staging_allocators.size(); ++i) {
			if (!staging_allocators[i]) {
				staging_allocators[i] = allocator;
				return;
			}
		}
	}

	static inline bool string_ends_with_fp16p(const char* name)
	{
		int len = strlen(name);
		if (len < 6)
			return false;

		return memcmp(name + len - 6, "_fp16p", 6) == 0;
	}

	static inline bool string_ends_with_fp16s(const char* name)
	{
		int len = strlen(name);
		if (len < 6)
			return false;

		return memcmp(name + len - 6, "_fp16s", 6) == 0;
	}

	static inline bool string_ends_with_fp16a(const char* name)
	{
		int len = strlen(name);
		if (len < 6)
			return false;

		return memcmp(name + len - 6, "_fp16a", 6) == 0;
	}

	int Device::create_shader_module() {
		shader_modules.resize(layer_shader_registry_entry_count, VK_NULL_HANDLE);
		for (int i = 0; i < layer_shader_registry_entry_count; ++i) {
			const char* shader_name = layer_shader_registry[i].name;
			if (!info.support_fp16_packed) {
				if (string_ends_with_fp16p(shader_name))
					continue;
			}

			if (!info.support_fp16_storage)	{
				if (string_ends_with_fp16s(shader_name))
					continue;
			}

			if (!info.support_fp16_arithmetic) {
				if (string_ends_with_fp16a(shader_name))
					continue;
			}

			VkShaderModule shader_module = compile_shader_module(layer_shader_registry[i].spv_data, layer_shader_registry[i].spv_data_size);
			if (shader_module == 0) return -1;
			shader_modules[i] = shader_module;
		}

		return 0;
	}

	void Device::destroy_shader_module() {
		for(int i =0; i < (int)shader_modules.size(); ++i)
			vkDestroyShaderModule(device, shader_modules[i], 0);
		shader_modules.clear();
	}


	int Device::init_device_extension()
	{
		if (info.support_VK_KHR_descriptor_update_template)
		{
			vkCreateDescriptorUpdateTemplateKHR = (PFN_vkCreateDescriptorUpdateTemplateKHR)vkGetDeviceProcAddr(device, "vkCreateDescriptorUpdateTemplateKHR");
			vkDestroyDescriptorUpdateTemplateKHR = (PFN_vkDestroyDescriptorUpdateTemplateKHR)vkGetDeviceProcAddr(device, "vkDestroyDescriptorUpdateTemplateKHR");
			vkUpdateDescriptorSetWithTemplateKHR = (PFN_vkUpdateDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(device, "vkUpdateDescriptorSetWithTemplateKHR");
		}

		if (info.support_VK_KHR_get_memory_requirements2)
		{
			vkGetImageMemoryRequirements2KHR = (PFN_vkGetImageMemoryRequirements2KHR)vkGetDeviceProcAddr(device, "vkGetImageMemoryRequirements2KHR");
			vkGetBufferMemoryRequirements2KHR = (PFN_vkGetBufferMemoryRequirements2KHR)vkGetDeviceProcAddr(device, "vkGetBufferMemoryRequirements2KHR");
			vkGetImageSparseMemoryRequirements2KHR = (PFN_vkGetImageSparseMemoryRequirements2KHR)vkGetDeviceProcAddr(device, "vkGetImageSparseMemoryRequirements2KHR");
		}

		if (info.support_VK_KHR_push_descriptor)
		{
			if (info.support_VK_KHR_descriptor_update_template)
			{
				vkCmdPushDescriptorSetWithTemplateKHR = (PFN_vkCmdPushDescriptorSetWithTemplateKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetWithTemplateKHR");
			}

			vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");
		}

		return 0;
	}

	Device* get_gpu_device(int device_index) {
		if (device_index < 0 || device_index >= device_count)
			return 0;

		MutexLockGuard lock(default_dev_lock);

		if (!default_dev[device_index])
			default_dev[device_index] = new Device(device_index);

		return default_dev[device_index];
	}
}