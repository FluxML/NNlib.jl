# This should probably be its own package

_cufunc(f,x) = f
_cufunc(f,x,xs...) = _cufunc(f, xs...)

using MacroTools

isbcastop(x) = isexpr(x, :call) && x.args[1] in :[.*,./,.+,.-].args

broadcast_inputs(ex) =
  ex isa Symbol ? [ex] :
  @capture(ex, f_.(args__)) ? vcat(broadcast_inputs.(args)...) :
  isbcastop(ex) ? vcat(broadcast_inputs.(ex.args[2:end])...) :
  []

macro fix(ex)
  fs = []
  ex = MacroTools.prewalk(ex) do ex
    @capture(ex, f_.(args__)) || return ex
    # May not work in cases like `x .+ log(1.0)` but w/e
    xs = broadcast_inputs(ex)
    f_ = gensym()
    push!(fs, :($f_ = $_cufunc($f, $(xs...))))
    :($f_.($(args...)))
  end
  :($(fs...); $ex) |> esc
end
