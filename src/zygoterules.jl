# # This is a performance hack specifically for Zygote, because it doesn't handle fused
# # broadcasts well
# for (f, df) in [
#     (:relu, :(x .> 0)),
#     (:selu, :(dselu.(x))),
#     (:elu, :(delu.(x))),
#     (:σ, :(conj.(Ω .* (1 .- Ω)))),
# ]
#     pullback = Symbol(:broadcasted_, f, :_pullback)
#     @eval @adjoint function Base.Broadcast.broadcasted(::typeof($f), x::Numeric)
#         Ω = $f.(x)
#         $pullback(Δ) = (nothing, Δ .* $df)
#         return Ω, $pullback
#     end
# end

# This is a performance hack specifically for Zygote, because it doesn't handle fused
# broadcasts well
using  Base.Broadcast: broadcasted

for (f, df) in [
    (:relu, :(x .> 0)),
    (:selu, :(dselu.(x))),
    (:elu, :(delu.(x))),
    (:σ, :(conj.(Ω .* (1 .- Ω)))),
    ]

    pullback = Symbol(:broadcasted_, f, :_pullback)
    @eval function ChainRulesCore.rrule(::typeof(broadcasted),
                                                ::typeof($f), x::Numeric)
        Ω = $f.(x)
        $pullback(Δ) = (NO_FIELDS, NO_FIELDS, Δ .* $df)
        return Ω, $pullback
    end
end
