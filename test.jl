import Metal, NNlib, Flux

dev = Flux.get_device()

src, idx = Int32[1 2 3 4; 5 6 7 8], Int32[2,1,1,5]
srcd, idxd = dev(x), dev(idx)
y = NNlib.scatter(+, src, idx)
yd = dev(zero(y))
NNlib.scatter!(+, yd, srcd, idxd)


