using Documenter, NNlib

DocMeta.setdocmeta!(NNlib, :DocTestSetup, :(using NNlib); recursive = true)

makedocs(modules = [NNlib],
         sitename = "NNlib.jl",
         doctest = false,
         pages = ["Home" => "index.md",
                  "Reference" => "reference.md"],
         format = Documenter.HTML(
              canonical = "https://fluxml.ai/NNlib.jl/stable/",
            #   analytics = "UA-36890222-9",
              assets = ["assets/flux.css"],
              prettyurls = get(ENV, "CI", nothing) == "true"),
              warnonly=[:missing_docs,]
        )

deploydocs(repo = "github.com/FluxML/NNlib.jl.git",
           target = "build",
           push_preview = true)
