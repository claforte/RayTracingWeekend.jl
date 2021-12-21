"Axis-Aligned Bounding Box"
struct Aabb{T}
    min::Vec3{T} # minimum component along each axis
    max::Vec3{T} # max along each axis
end