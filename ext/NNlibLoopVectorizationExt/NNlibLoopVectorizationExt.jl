module NNlibLoopVectorizationExt

using NNlib
using LoopVectorization
using Random, Statistics
using OffsetArrays, Static

# Bevor die Methoden überladen werden, sollte ein Selfcheck durchgeführt werden, ob die Ergebnisse mit NNlib übereinstimmen.
# Wenn nicht, sollte eine Warnung angezeigt werden und die wenn irgendwie möglich, nur die funktionierenden Methoden verwendet werden.
# Z.b. in dem bei falschem Ergebniss, die im2col Methode als Backend in der überladenen Methode aufgerufen wird.

include("conv.jl")
include("pooling.jl")

end # module