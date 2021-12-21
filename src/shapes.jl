struct Sphere{T <: AbstractFloat} <: Hittable
	center::Vec3{T}
	radius::T
	mat::Material{T}
end

struct MovingSphere{T <: AbstractFloat} <: Hittable
	center0::Vec3{T} # center position at time time0
    center1::Vec3{T} # center position at time time1
    time0::T
    time1::T
	radius::T
	mat::Material{T}
end

"Evaluate a Sphere at specified time"
sphere(ms::MovingSphere{T}, time::T) where T = Sphere(
    ms.center0+((time-ms.time0))*(ms.center1-ms.center0), ms.radius, ms.mat)

