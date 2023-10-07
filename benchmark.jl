using NNlib, Flux, Metalhead
using BenchmarkTools
using DataFrames, CSV

forward(model, input) = model(input)

dummy_loss(output) = sum(output .- 1)

function train_step(model, input)
    ∇model, ∇input = gradient(model, input) do m, x
        dummy_loss(m(x))
    end
    return ∇model, ∇input
end

function benchmark(models, dtype, batch_sizes, channels, spatial_size)
    model_names = sort(collect(keys(models))) # make sure the models are always in the same order
    forward_times = zeros(length(model_names), length(batch_sizes))
    train_step_times = zeros(length(model_names), length(batch_sizes))

    for (i, model_name) in enumerate(model_names)
        println("Benchmarking $model_name...")
        for (j, batch_size) in enumerate(batch_sizes)
    
            input = rand(dtype, spatial_size..., channels, batch_size)
            model = models[model_name]
    
            forward(model, input) # compilation
            train_step(model, input) # compilation

            forward_times[i, j] = @belapsed forward($model, $input) # @elapsed
            train_step_times[i, j] = @belapsed train_step($model, $input) # @elapsed

        end
    end

    return forward_times, train_step_times
end

# models which should be benchmarked
models = Dict(
    "ResNet18" => ResNet(18),
    "WideResNet50" => WideResNet(50),
    "DenseNet121" => DenseNet(121),
    "EfficientNet" => EfficientNet(:b0),
    "EfficientNetv2" => EfficientNetv2(:small),
    "MobileNetv3" => MobileNetv3(:small),
    # "GoogLeNet" => GoogLeNet(),
    "ConvNeXt" => ConvNeXt(:tiny),
)

# the data type and batch sizes which should be benchmarked
dtype = Float32
batch_sizes = (1, 32)
# size information (e.g. ImageNet-like images)
channels = 3
spatial_size = (224, 224) # WH

forward_times1, train_step_times1 = benchmark(models, dtype, batch_sizes, channels, spatial_size)
using LoopVectorization # load LoopVectorization here to load the lv-extension
forward_times2, train_step_times2 = benchmark(models, dtype, batch_sizes, channels, spatial_size)

df = DataFrame()
df[!, "model_names"] = sort(collect(keys(models))) # make sure the models are always in the same order

for (i, batch_size) in enumerate(batch_sizes)
    df[!, "acceleration inference, batch_size: $batch_size"] = forward_times1[:, i] ./ forward_times2[:, i]
    df[!, "acceleration train, batch_size: $batch_size"] = train_step_times1[:, i] ./ train_step_times2[:, i]

    df[!, "im2col, inference, batch_size: $batch_size"] = forward_times1[:, i]
    df[!, "lv-ext, inference, batch_size: $batch_size"] = forward_times2[:, i]
    df[!, "im2col, train, batch_size: $batch_size"] = train_step_times1[:, i]
    df[!, "lv-ext, train, batch_size: $batch_size"] = train_step_times2[:, i]
end

CSV.write("benchmark_result_julia.csv", df)