"""
    safe_div(x, y)

Safely divde `x` by `y`. If `y` is zero, return `x` directly.
"""
safe_div(x, y) = ifelse(iszero(y), x, x/y)
