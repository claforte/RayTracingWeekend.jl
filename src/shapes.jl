struct Sphere{T <: AbstractFloat} <: Hittable
	center::Vec3{T}
	radius::T
	mat::Material{T}
end

struct MovingSphere{T <: AbstractFloat} <: Hittable
	center0::Vec3{T}
    center1::Vec3{T}
	radius::T
	mat::Material{T}
end