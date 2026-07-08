from torchvision import transforms

def get_train_transform(image_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)), # Make every image the same size (224×224)
        transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation by flipping images
        transforms.RandomRotation(degrees=10), # Small rotations improve robustness
        transforms.ToTensor(), # Convert PIL image → PyTorch Tensor 
        transforms.Normalize( # Normalize using ImageNet statistics -> better with pre-trained models
            mean = [0.485,0.456,0.406],
            std = [0.229,0.224,0.225]
        )
    ])

    return train_transform

def get_test_transform(image_size=224): # No random augmentations here
    test_transform = transforms.Compose([
        transforms.Resize((image_size,image_size)), # Make every image the same size (224×224)
        transforms.ToTensor(), # Convert PIL image → PyTorch Tensor 
        transforms.Normalize( # Normalize using ImageNet statistics -> better with pre-trained models
            mean = [0.485,0.456,0.406],
            std = [0.229,0.224,0.225]
        )
    ])

    return test_transform