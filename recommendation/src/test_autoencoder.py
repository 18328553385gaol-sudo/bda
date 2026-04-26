import torch
from autoencoder import PlayerAutoEncoder

def main():
    input_dim = 12  # 你的 feature 数量
    model = PlayerAutoEncoder(input_dim=input_dim, latent_dim=4)

    # 随机造一批数据（模拟球员特征）
    x = torch.randn(10, input_dim)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()