import torch


def main():
    print(f"Torch: {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA: {cuda}")

    if cuda:
        device_count = torch.cuda.device_count()
        print(f"Number of GPUs: {device_count}")
    else:
        print("No GPU")


if __name__ == "__main__":
    main()
