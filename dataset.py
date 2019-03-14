import torch
from torchvision import transforms, datasets


def __mnist_data():
    compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    outdir = './data'

    return datasets.MNIST(root=outdir, train=True, transform=compose, download=True)


def get_dataset():
    data = __mnist_data()

    dataloader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

    num_batches = len(dataloader)

    return dataloader, num_batches


def images_to_vectors(images):
    return images.view(images.size(0), 784)


def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28).permute(0, 2, 3, 1)