import torch

if __name__ =="__main__":
    if (torch.cuda.is_available()):
        cuda_version = torch.version.cuda
        print(f'CUDA: {cuda_version}')
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            print(f'ID {i}: {torch.cuda.get_device_name(i)}')
    else:
        print("CUDA unavailable.")

