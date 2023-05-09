# Adapted from
# https://github.com/kul-forbes/ProximalOperators.jl/tree/master/benchmark
using ArgParse
using PkgBenchmark
using BenchmarkCI: displayjudgement, printresultmd, CIResult
using Markdown

function markdown_report(judgement)
    md = sprint(printresultmd, CIResult(judgement = judgement))
    md = replace(md, ":x:" => "❌")
    md = replace(md, ":white_check_mark:" => "✅")
    return md
end

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--target"
            help = "the branch/commit/tag to use as target"
            default = "HEAD"
        "--baseline"
            help = "the branch/commit/tag to use as baseline"
            default = "master"
        "--retune"
            help = "force re-tuning (ignore existing tuning data)"
            action = :store_false
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()

    mkconfig(; kwargs...) =
        BenchmarkConfig(
            env = Dict(
                "JULIA_NUM_THREADS" => get(ENV, "JULIA_NUM_THREADS", "1"),
            );
            kwargs...
        )

    target = parsed_args["target"]
    group_target = benchmarkpkg(
        dirname(@__DIR__),
        mkconfig(id = target),
        resultfile = joinpath(@__DIR__, "result-$(target).json"),
        retune = parsed_args["retune"],
    )

    baseline = parsed_args["baseline"]
    group_baseline = benchmarkpkg(
        dirname(@__DIR__),
        mkconfig(id = baseline),
        resultfile = joinpath(@__DIR__, "result-$(baseline).json"),
    )

    judgement = judge(group_target, group_baseline)
    report_md = markdown_report(judgement)
    write(joinpath(@__DIR__, "report.md"), report_md)
    display(Markdown.parse(report_md))
end

main()
