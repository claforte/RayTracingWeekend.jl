# basic structs, e.g. Ray, Material, HitRecord, Scatter, Sphere
struct Ray{T}
	origin::Vec3{T} # Point 
	dir::Vec3{T} # Vec3 # direction (unit vector)
    time::T # Time "when" the ray exists (used for motion blur)
end

# claforte: TODO: replace by constructor
@inline function ray(origin::Vec3{T}=(SA{T}[0,0,0]), dir::Vec3{T}=(SA{T}[0,0,-1]), time::T=T(0)) where T
    Ray{T}(origin, dir, time)
end

"An object that can be hit by Ray"
abstract type Hittable end

const HittableList = Vector{Hittable}

"""Materials tell us how rays interact with a surface"""
abstract type Material{T <: AbstractFloat} end

"Record a hit between a ray and an object's surface"
struct HitRecord{T <: AbstractFloat}
	t::T # distance from the ray's origin to the intersection with a surface. 
	
	p::Vec3{T} # point of the intersection between an object's surface and a ray
	n⃗::Vec3{T} # surface's outward normal vector, points towards outside of object?
	
	# If true, our ray hit from outside to the front of the surface. 
	# If false, the ray hit from within.
	front_face::Bool
	
	mat::Material{T}

	@inline HitRecord(t::T,p,n⃗,front_face,mat) where T = new{T}(t,p,n⃗,front_face,mat)
end

struct Scatter{T<: AbstractFloat}
	r::Ray{T}
	attenuation::Vec3{T}
	
	# claforte: TODO: rename to "absorbed?", i.e. not reflected/refracted?
	reflected::Bool # whether the scattered ray was reflected, or fully absorbed
	@inline Scatter(r::Ray{T},a::Vec3{T},reflected=true) where T = new{T}(r,a,reflected)
end
