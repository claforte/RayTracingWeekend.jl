module RayTracingWeekend

using Images
using LinearAlgebra
using Random
using RandomNumbers.Xorshifts
using StaticArrays

export color_vec3_in_rgb, default_camera, get_ray, hit, near_zero, point, random_between, random_vec2, 
        random_vec2_in_disk, random_vec3, random_vec3_in_sphere, random_vec3_on_sphere, ray_color, ray_to_HitRecord, reflect, 
        reflectance, refract, render, reseed!, rgb, rgb_gamma2, scatter, skycolor, squared_length, trand
export Camera, Dielectric, Hittable, HittableList, HitRecord, Lambertian, Material, Metal, Ray, Scatter, Sphere, Vec3
export scene_2_spheres, scene_4_spheres, scene_blue_red_spheres, scene_diel_spheres, scene_random_spheres
export TRNG

include("vec.jl")

struct Ray{T}
	origin::Vec3{T} # Point 
	dir::Vec3{T} # Vec3 # direction (unit vector)
end

@inline point(r::Ray{T}, t::T) where T <: AbstractFloat = r.origin .+ t .* r.dir # equivalent to C++'s ray.at()

@inline function skycolor(ray::Ray{T}) where T
	white = SA[1.0, 1.0, 1.0]
	skyblue = SA[0.5, 0.7, 1.0]
	t = T(0.5)*(ray.dir.y + one(T))
    (one(T)-t)*white + t*skyblue
end

# Per-thread Random Number Generator. Initialized later...
const TRNG = Xoroshiro128Plus[]

function __init__()
	# Instantiate 1 RNG (Random Number Generator) per thread, for performance.
	# This can't be done during precompilation since the number of threads isn't known then.
	resize!(TRNG, Threads.nthreads())
	for i in 1:Threads.nthreads()
		TRNG[i] = Xoroshiro128Plus(i)
	end
	nothing
end

include("rand.jl")

"An object that can be hit by Ray"
abstract type Hittable end

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

struct Sphere{T <: AbstractFloat} <: Hittable
	center::Vec3{T}
	radius::T
	mat::Material{T}
end

"""Equivalent to `hit_record.set_face_normal()`"""
@inline @fastmath function ray_to_HitRecord(t::T, p, outward_n⃗, r_dir::Vec3{T}, mat::Material{T})::Union{HitRecord,Nothing} where T
	front_face = r_dir ⋅ outward_n⃗ < 0
	n⃗ = front_face ? outward_n⃗ : -outward_n⃗
	HitRecord(t,p,n⃗,front_face,mat)
end

struct Scatter{T<: AbstractFloat}
	r::Ray{T}
	attenuation::Vec3{T}
	
	# claforte: TODO: rename to "absorbed?", i.e. not reflected/refracted?
	reflected::Bool # whether the scattered ray was reflected, or fully absorbed
	@inline Scatter(r::Ray{T},a::Vec3{T},reflected=true) where T = new{T}(r,a,reflected)
end

"""Compute reflection vector for v (pointing to surface) and normal n⃗.

	See [diagram](https://raytracing.github.io/books/RayTracingInOneWeekend.html#metal/mirroredlightreflection)"""
@inline @fastmath reflect(v::Vec3{T}, n⃗::Vec3{T}) where T = v - (2v⋅n⃗)*n⃗

@inline @fastmath function hit(s::Sphere{T}, r::Ray{T}, tmin::T, tmax::T)::Union{HitRecord,Nothing} where T
    oc = r.origin - s.center
    #a = r.dir ⋅ r.dir # unnecessary since `r.dir` is normalized
	a = 1
	half_b = oc ⋅ r.dir
    c = oc⋅oc - s.radius^2
    discriminant = half_b^2 - a*c
	if discriminant < 0 return nothing end # no hit!
	sqrtd = √discriminant
	
	# Find the nearest root that lies in the acceptable range
	root = (-half_b - sqrtd) / a	
	if root < tmin || tmax < root
		root = (-half_b + sqrtd) / a
		if root < tmin || tmax < root
			return nothing # no hit!
		end
	end
	
	t = root
	p = point(r, t)
	n⃗ = (p - s.center) / s.radius
	return ray_to_HitRecord(t, p, n⃗, r.dir, s.mat)
end

const HittableList = Vector{Hittable}

"""Find closest hit between `Ray r` and a list of Hittable objects `h`, within distance `tmin` < `tmax`"""
@inline function hit(hittables::HittableList, r::Ray{T}, tmin::T, tmax::T)::Union{HitRecord,Nothing} where T
    closest = tmax # closest t so far
    best_rec::Union{HitRecord,Nothing} = nothing # by default, no hit
    @inbounds for i in eachindex(hittables) # @paulmelis reported gave him a 4X speedup?!
		h = hittables[i]
        rec = hit(h, r, tmin, closest)
        if rec !== nothing
            best_rec = rec
            closest = best_rec.t # i.e. ignore any further hit > this one's.
        end
    end
    best_rec
end

@inline color_vec3_in_rgb(v::Vec3{T}) where T = 0.5normalize(v) + SA{T}[0.5,0.5,0.5]


"""Compute color for a ray, recursively

	Args:
		depth: how many more levels of recursive ray bounces can we still compute?"""
@inline @fastmath function ray_color(r::Ray{T}, world::HittableList, depth=16) where T
    if depth <= 0
		return SA{T}[0,0,0]
	end
		
	rec::Union{HitRecord,Nothing} = hit(world, r, T(1e-4), typemax(T))
    if rec !== nothing
		# For debugging, represent vectors as RGB:
		# claforte TODO: adapt to latest code!
		# return color_vec3_in_rgb(rec.p) # show the normalized hit point
		# return color_vec3_in_rgb(rec.n⃗) # show the normal in RGB
		# return color_vec3_in_rgb(rec.p + rec.n⃗)
		# return color_vec3_in_rgb(random_vec3_in_sphere())
		#return color_vec3_in_rgb(rec.n⃗ + random_vec3_in_sphere())

        s = scatter(rec.mat, r, rec)
		if s.reflected
			return s.attenuation .* ray_color(s.r, world, depth-1)
		else
			return SA{T}[0,0,0]
		end
    else
        skycolor(r)
    end
end

include("camera.jl")

"""Render an image of `scene` using the specified camera, number of samples.

	Args:
		scene: a HittableList, e.g. a list of spheres
		n_samples: number of samples per pixel, eq. to C++ samples_per_pixel

	Equivalent to C++'s `main` function."""
function render(scene::HittableList, cam::Camera{T}, image_width=400,
				n_samples=1) where T
	# Image
	aspect_ratio = T(16.0/9.0) # TODO: use cam.aspect_ratio for consistency
	image_height = convert(Int64, floor(image_width / aspect_ratio))

	# Render
	img = zeros(RGB{T}, image_height, image_width)
	f32_image_width = convert(Float32, image_width)
	f32_image_height = convert(Float32, image_height)
	
	# Reset the random seeds, so we always get the same images...
	# Makes comparing performance more accurate.
	reseed!()

	Threads.@threads for i in 1:image_height
		@inbounds for j in 1:image_width # iterate over each row (FASTER?!)
			accum_color = SA{T}[0,0,0]
			u = convert(T, j/image_width)
			v = convert(T, (image_height-i)/image_height) # i is Y-down, v is Y-up!
			
			for s in 1:n_samples
				if s == 1 # 1st sample is always centered
					δu = δv = T(0)
				else
					# Supersampling antialiasing.
					δu = trand(T) / f32_image_width
					δv = trand(T) / f32_image_height
				end
				ray = get_ray(cam, u+δu, v+δv)
				accum_color += ray_color(ray, scene)
			end
			img[i,j] = rgb_gamma2(accum_color / n_samples)
		end
	end
	img
end

"""
	Args:
		refraction_ratio: incident refraction index divided by refraction index of 
			hit surface. i.e. η/η′ in the figure above"""
@inline @fastmath function refract(dir::Vec3{T}, n⃗::Vec3{T}, refraction_ratio::T) where T
	cosθ = min(-dir ⋅ n⃗, one(T))
	r_out_perp = refraction_ratio * (dir + cosθ*n⃗)
	r_out_parallel = -√(abs(one(T)-squared_length(r_out_perp))) * n⃗
	normalize(r_out_perp + r_out_parallel)
end

@inline @fastmath function reflectance(cosθ, refraction_ratio)
	# Use Schlick's approximation for reflectance.
	# claforte: may be buggy? I'm getting black pixels in the Hollow Glass Sphere...
	r0 = (1-refraction_ratio) / (1+refraction_ratio)
	r0 = r0^2
	r0 + (1-r0)*((1-cosθ)^5)
end

include("material.jl")

include("scenes.jl")

end
