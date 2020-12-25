"""
    save_div(x, y)

Savely divde `x` by `y`. If `y` is zero, return `x` directly.
"""
save_div(x, y) = ifelse(iszero(y), x, x/y)
