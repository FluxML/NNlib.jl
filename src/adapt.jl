# This doesn't really belong here, but it's convenient.

adapt_(T, x) = x

adapt(T, x) = adapt_(T, x)

adapt(T, x::RowVector) = RowVector(adapt(T, x.vec))
