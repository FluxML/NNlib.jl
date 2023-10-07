import torch
import torchvision.models as visionmodels
import time
import pandas as pd

def dummy_loss(output):
    return torch.sum(output - 1)

def train_step(model, input_to_model):
    output = model(input_to_model)
    loss = dummy_loss(output)
    loss.backward()

def benchmark(models, batch_sizes, channels, spatial_size):
    model_names = sorted(list(models.keys())) # make sure the models are always in the same order
    forward_times = torch.zeros(len(model_names), len(batch_sizes))
    train_step_times = torch.zeros(len(model_names), len(batch_sizes))

    for i, model_name in enumerate(model_names):
        print(f"Benchmarking {model_name}...")
        for j, batch_size in enumerate(batch_sizes):
    
            input_to_model = torch.rand(batch_size, channels, spatial_size[0], spatial_size[1])
            model = models[model_name]
    
            time_start = time.perf_counter()
            model(input_to_model)
            time_duration = time.perf_counter() - time_start
            forward_times[i, j] = time_duration

            time_start = time.perf_counter()
            train_step(model, input_to_model)
            time_duration = time.perf_counter() - time_start
            train_step_times[i, j] = time_duration

    return forward_times, train_step_times

models = {
    "ResNet18" : visionmodels.resnet18(),
    "WideResNet50" : visionmodels.wide_resnet50_2(),
    "DenseNet121" : visionmodels.densenet121(),
    "EfficientNet" : visionmodels.efficientnet_b0(),
    "EfficientNetv2" : visionmodels.efficientnet_v2_s(),
    "MobileNetv3" : visionmodels.mobilenet_v3_small(),
    # "GoogLeNet" : visionmodels.googlenet(),
    "ConvNeXt" : visionmodels.convnext_tiny(),
}

# the batch sizes which should be benchmarked
batch_sizes = (1, 32)
# size information (e.g. ImageNet-like images)
channels = 3
spatial_size = (224, 224) # HW

forward_times, train_step_times = benchmark(models, batch_sizes, channels, spatial_size)

df = pd.DataFrame()
df["model_names"] = sorted(list(models.keys())) # make sure the models are always in the same order

for (i, batch_size) in enumerate(batch_sizes):
    df[f"inference, batch_size: {batch_size}"] = forward_times[:, i]
    df[f"train, batch_size: {batch_size}"] = train_step_times[:, i]

df.to_csv("benchmark_result_pytorch.csv")