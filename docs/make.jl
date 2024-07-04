using Documenter, NNlib

DocMeta.setdocmeta!(NNlib, :DocTestSetup,
    :(using FFTW, NNlib, UnicodePlots); recursive = true)

makedocs(modules = [NNlib],
    sitename = "NNlib.jl",
    doctest = true,
    pages = ["Home" => "index.md",
             "Reference" => "reference.md",
             "Audio" => "audio.md"],
    format = Documenter.HTML(
        canonical = "https://fluxml.ai/NNlib.jl/stable/",
        # analytics = "UA-36890222-9",
        assets = ["assets/flux.css"],
        prettyurls = get(ENV, "CI", nothing) == "true"),
    warnonly=[:missing_docs,]
)

deploydocs(repo = "github.com/FluxML/NNlib.jl.git",
           target = "build",
           push_preview = true)
