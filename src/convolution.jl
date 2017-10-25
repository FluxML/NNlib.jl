using ImageFiltering

function conv(x, w)
  @assert(size(x, ndims(w)-1) == size(w, ndims(w)-1))
  filters = size(w, ndims(w))
  cs = map(1:filters) do f
    w′ = reflect(centered(slicedim(w, ndims(w), f)))
    imfilter(x, w′, Inner()).parent
  end
  cat(ndims(w)-1, cs...)
end
