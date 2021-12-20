# Operations on 2D and 3D vectors, and colors

const Vec3{T<:AbstractFloat} = SVector{3, T}

# Adapted from @Christ_Foster's: https://discourse.julialang.org/t/ray-tracing-in-a-week-end-julia-vs-simd-optimized-c/72958/40
# Christ added this feature directly in StaticArrays.jl on 2021-12-17, so 
# this will soon be no longer needed.
@inline @inbounds function Base.getproperty(v::SVector{3}, name::Symbol)
    name === :x || name === :r ? getfield(v, :data)[1] :
    name === :y || name === :g ? getfield(v, :data)[2] :
    name === :z || name === :b ? getfield(v, :data)[3] : getfield(v, name)
end

@inline @inbounds function Base.getproperty(v::SVector{2}, name::Symbol)
    name === :x || name === :r ? getfield(v, :data)[1] :
    name === :y || name === :g ? getfield(v, :data)[2] : getfield(v, name)
end

@inline squared_length(v::Vec3) = v ⋅ v
@inline near_zero(v::Vec3) = squared_length(v) < 1e-5
@inline rgb(v::Vec3) = RGB(v...)
@inline rgb_gamma2(v::Vec3) = RGB(sqrt.(v)...)