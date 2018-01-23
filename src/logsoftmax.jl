using Base.Threads

function logsoftmax!(out::AbstractVecOrMat, xs::AbstractVecOrMat)
  @threads for j = 1:size(xs, 2)
    @inbounds begin
      xi_max = xs[1, j]
      for i = 1:size(out, 1)
        xi_max = max(xi_max, xs[i, j])
      end
      s = zero(eltype(out))
      for i = 1:size(out, 1)
        s += exp(xs[i, j] - xi_max)
      end
      for i = 1:size(out, 1)
        out[i, j] = xs[i, j] - log(s) - xi_max
      end
    end
  end
  out
end


logsoftmax!(xs) = logsoftmax!(xs, xs)
logsoftmax(xs) = logsoftmax!(similar(xs), xs)

∇logsoftmax(Δ, xs) = ∇softmax(Δ ./ softmax(xs), xs)

"""
    logsoftmax(xs) = log.(exp.(xs) ./ sum(exp.(xs)))

logsoftmax computes the log of softmax(xs) and it is more numerically stable
than softmax function in computing the cross entropy loss.
"""
logsoftmax
