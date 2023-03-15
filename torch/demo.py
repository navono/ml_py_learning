import torch

if __name__ == "__main__":
    print(torch.rand(5, 3))
    print(torch.cuda.is_available())
