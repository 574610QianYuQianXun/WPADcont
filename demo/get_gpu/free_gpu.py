import pynvml

def get_max_free_memory_gpu():
    # 初始化 NVML
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    max_free_mem = 0
    best_gpu_index = 0

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mem = mem_info.free

        print(f"GPU {i}: Free Memory: {free_mem / 1024**2:.2f} MB")

        if free_mem > max_free_mem:
            max_free_mem = free_mem
            best_gpu_index = i

    pynvml.nvmlShutdown()
    return best_gpu_index

if __name__ == "__main__":
    best_gpu = get_max_free_memory_gpu()
    print(f"GPU with the most free memory is: GPU {best_gpu}")
