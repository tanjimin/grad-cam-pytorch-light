import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from grad_cam import grad_cam

def main():
    boxer_example()
    tiger_cat_example()
    elephant_example()

def boxer_example():
    model = torchvision.models.resnet34(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    heatmap_layer = model.layer4[2].conv2
    image = Image.open("./images/cat_dog.png")
    input_tensor = transform(image)
    boxer_label = 242
    image = grad_cam(model, input_tensor, heatmap_layer, boxer_label)
    plt.imshow(image)
    plt.savefig('./images/boxer_grad-cam')

def tiger_cat_example():
    model = torchvision.models.resnet34(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    heatmap_layer = model.layer4[2].conv2
    image = Image.open("./images/cat_dog.png")
    input_tensor = transform(image)
    tiger_cat_label = 282
    image = grad_cam(model, input_tensor, heatmap_layer, tiger_cat_label)
    plt.imshow(image)
    plt.savefig('./images/tiger_cat_grad-cam')

def elephant_example():
    model = torchvision.models.resnet34(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    heatmap_layer = model.layer4[2].conv2
    image = Image.open("./images/elephant.jpg")
    input_tensor = transform(image)
    elephant_label = 386
    image = grad_cam(model, input_tensor, heatmap_layer, elephant_label)
    plt.imshow(image)
    plt.savefig('./images/elephant_grad-cam')

if __name__== "__main__":
    main()