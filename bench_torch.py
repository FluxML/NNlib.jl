import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import time

model = models.efficientnet_v2_m()
model.eval()

b_size = 1
img = torch.rand(b_size, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
    with record_function("model_inference"):
        pred = model(img)
    """
    with record_function("model_backward"):
        loss = torch.sum(pred - 0.5) # dummy loss
        loss.backward()
    """

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=-1))
# print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=-1))

start1 = time.perf_counter()
pred = model(img)
start2 = time.perf_counter()
loss = torch.sum(pred - 0.5) # dummy loss
loss.backward()
end = time.perf_counter()
print(f"Time used inference: {start2 - start1} seconds")
print(f"Time used backward: {end - start2} seconds")
print(f"Time used inference and backward: {end - start1} seconds")