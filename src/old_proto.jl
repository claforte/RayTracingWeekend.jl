# Prototype - copied from pluto_RayTracingWeekend.jl
# Adapted from [Ray Tracing In One Weekend by Peter Shirley](https://raytracing.github.io/books/RayTracingInOneWeekend.html) and 
# [cshenton's Julia implementation](https://github.com/cshenton/RayTracing.jl)"

using BenchmarkTools, Images, InteractiveUtils, LinearAlgebra, StaticArrays
using LoopVectorization: @turbo
#using SIMD: @simd

Threads.nthreads()

const Vec3{T<:AbstractFloat} = SVector{3, T}
t_col = SA[0.4, 0.5, 0.1] # test color. See StaticArrays.jl

# Adapted from @Christ_Foster's: https://discourse.julialang.org/t/ray-tracing-in-a-week-end-julia-vs-simd-optimized-c/72958/40
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

# Before optimization:
#   677-699 ns (41 allocations: 2.77 KiB) # squared_length(v::SVector) = v ⋅ v; @btime squared_length(t_col)
# inside a function:
#   894-906 ns (43 allocations: 2.83 KiB)
#
# don't define `Base.getproperty(vec::SVector{3}, sym::Symbol)`
#   208.527 ns (3 allocations: 80 bytes)
#
# just test `@btime squared_length(t_col)`:
#    12.115 ns (1 allocation: 16 bytes)
# test again after redefining: `Vec3 = SVector{3, Float64}` instead of `Vec3 = SVector{3}`
#    11.966 ns (1 allocation: 16 bytes)
# test again after redefining: `Vec3 = SVector{3, Float32}` instead of `Vec3 = SVector{3, Float64}`
#    13.652 ns (1 allocation: 16 bytes)
#    (SLOWER! I wonder if this would hold with larger vectors or if we become mem-bound) 
# replace: squared_length(v::SVector{3,MyFloat})
#     3.256 ns (0 allocations: 0 bytes)
# replace: @btime squared_length($t_col) # notice the $
#     1.152 ns (0 allocations: 0 bytes)
# use `const Vec3{T<:AbstractFloat} = SVector{3, T}`:
#     1.162 ns (0 allocations: 0 bytes) (i.e. equivalent)
#@btime squared_length($t_col)

@inline near_zero(v::Vec3) = squared_length(v) < 1e-5
#@btime near_zero($t_col) # 1.382 ns (0 allocations: 0 bytes)

# Test images

img = rand(4, 3)

img_rgb = rand(RGB, 4, 4)

ex1 = [RGB(1,0,0) RGB(0,1,0) RGB(0,0,1);
       RGB(1,1,0) RGB(1,1,1) RGB(0,0,0)]

ex2 = zeros(RGB, 2, 3)
ex2[1,1] = RGB(1,0,0)
ex2

# The "Hello World" of graphics
@fastmath function gradient(nx::Int, ny::Int)
	img = zeros(RGB, ny, nx)
	@inbounds @simd for j in 1:nx
		for i in 1:ny # Julia is column-major, i.e. iterate 1 column at a time
			x = j; y = (ny-i);
			r = x/nx
			g = y/ny
			b = 0.2
			img[i,j] = RGB(r,g,b)
		end
	end
	img
end


# no explicit SIMD:
#  115.968 μs (20002 allocations: 781.33 KiB)
# w/ @turbo:
#  106.090 μs (20009 allocations: 781.61 KiB)
# w/ just @inbounds:
#  102.674 μs (20002 allocations: 781.33 KiB)
# w/ @fastmath, @inbounds, @simd:
#  100.209 μs (20002 allocations: 781.33 KiB)
gradient(200,100)

@inline rgb(v::Vec3) = RGB(v...)
#rgb($t_col) # 1.172 ns (0 allocations: 0 bytes)

@inline rgb_gamma2(v::Vec3) = RGB(sqrt.(v)...)
#rgb_gamma2($t_col) # 3.927 ns (0 allocations: 0 bytes)

struct Ray{T}
	origin::Vec3{T} # Point 
	dir::Vec3{T} # Vec3 # direction (unit vector)
end

_origin = SA[0.0,0.0,0.0]
_v3_minusY = SA[0.0,-1.0,0.0]
_t_ray1 = Ray(_origin, _v3_minusY)
# Float32: 6.161 ns (0 allocations: 0 bytes)
# Float64: 6.900 ns (0 allocations: 0 bytes)
#@btime Ray($_origin, $_v3_minusY) 

# equivalent to C++'s ray.at()
@inline point(r::Ray{T}, t::T) where T <: AbstractFloat = r.origin .+ t .* r.dir
#@btime point($_t_ray1, 0.5) # 1.412 ns (0 allocations: 0 bytes)

#md"# Chapter 4: Rays, simple camera, and background"

@inline function skycolor(ray::Ray{T}) where T
	white = SA[1.0, 1.0, 1.0]
	skyblue = SA[0.5, 0.7, 1.0]
	t = T(0.5)*(ray.dir.y + one(T))
    (one(T)-t)*white + t*skyblue
end
# 1.402 ns (0 allocations: 0 bytes)
#@btime skycolor($_t_ray1)

# before optim: 
#   1.474 μs (23 allocations: 608 bytes)
# pre-allocate args:
#   1.150 μs (19 allocations: 480 bytes)
# `rgb(skycolor(t_ray1))`
#   1.143 μs (19 allocations: 480 bytes)
# `skycolor(t_ray1)`:
#       636.259 ns (9 allocations: 256 bytes)
#    extract t_white and t_skyblue from skycolor():
#       279.779 ns (5 allocations: 128 bytes)
#    pre-allocate the ray given in argument:
#        51.130 ns (5 allocations: 128 bytes)
#    skycolor!($t_col2, $t_ray1, $t_white, $t_skyblue) using @set to return the value:
#        14.727 ns (1 allocation: 32 bytes)
#    replace all the Color() by @SVector[...], don't use global colors (use local)
#         1.402 ns (0 allocations: 0 bytes)
# restore the rgb(), i.e. `@btime rgb(skycolor($t_ray1))`
#   291.492 ns (4 allocations: 80 bytes)
# @inline a bunch of functions:
#     1.412 ns (0 allocations: 0 bytes)
#
#@btime rgb(skycolor($_t_ray1)) # 291.492 ns (4 allocations: 80 bytes)

# Instantiate 1 RNG (Random Number Generator) per thread, for performance
# Fix the random seeds, to make it easier to benchmark changes.
using Random, RandomNumbers.Xorshifts
const TRNG = [Xoroshiro128Plus(i) for i = 1:Threads.nthreads()]

reseed!() = for (i,rng) in enumerate(TRNG) Random.seed!(rng, i) end # reset the seed
reseed!()

@inline function trand() # thread-local rand()
    @inbounds rng = TRNG[Threads.threadid()]
    rand(rng)
end

@inline function trand(::Type{T}) where T # thread-local rand()
    @inbounds rng = TRNG[Threads.threadid()]
    rand(rng, T)
end
#@btime trand()

@inline random_between(min::T=0, max::T=1) where T = trand(T)*(max-min) + min # equiv to random_double()
# Float32: 4.519 ns (0 allocations: 0 bytes)
# Float64: 4.329 ns (0 allocations: 0 bytes)
# rand() using Xoroshiro128Plus _rng w/ Float64:
#    2.695 ns (0 allocations: 0 bytes)
#@btime random_between(50.0, 100.0) 

@inline random_vec3(min::T, max::T) where T = @inbounds @SVector[random_between(min, max) for i ∈ 1:3]

# Before optimization:
#   352.322 ns (6 allocations: 224 bytes)
# Don't use list comprehension:
#   179.684 ns (5 allocations: 112 bytes)
# Return @SVector[] instead of Vec3():
#    12.557 ns (0 allocations: 0 bytes)
# @inline lots of stuff:
#    10.440 ns (0 allocations: 0 bytes)
# Vec3{Float32}: 10.480 ns (0 allocations: 0 bytes)
# Vec3{Float64}: 10.882 ns (0 allocations: 0 bytes)
# rand() using MersenneTwister _rng w/ Float64:
#    3.787 ns (0 allocations: 0 bytes)
# rand() using Xoroshiro128Plus _rng w/ Float64:
#    4.448 ns (0 allocations: 0 bytes)
# using @inbounds:
#    4.298 ns (0 allocations: 0 bytes)
#@btime random_vec3(-1.0,1.0)

@inline random_vec2(min::T, max::T) where T = @inbounds @SVector[random_between(min, max) for i ∈ 1:2]
#@btime random_vec2(-1.0f0,1.0f0) # 3.677 ns (0 allocations: 0 bytes)

@inline function random_vec3_in_sphere(::Type{T}) where T # equiv to random_in_unit_sphere()
	while true
		p = random_vec3(T(-1), T(1))
		if p⋅p <= 1
			return p
		end
	end
end

# REMEMBER: the times are somewhat random! Use best timing of 5!
# Float32: 34.690 ns (0 allocations: 0 bytes)
# Float64: 34.065 ns (0 allocations: 0 bytes)
# rand() using MersenneTwister _rng w/ Float64:
#   21.333 ns (0 allocations: 0 bytes)
# rand() using Xoroshiro128Plus _rng w/ Float64:
#   19.716 ns (0 allocations: 0 bytes)
random_vec3_in_sphere(Float64)

"Random unit vector. Equivalent to C++'s `unit_vector(random_in_unit_sphere())`"
@inline random_vec3_on_sphere(::Type{T}) where T = normalize(random_vec3_in_sphere(T))

# REMEMBER: the times are somewhat random! Use best timing of 5! 
# Before optim:
#   517.538 ns (12 allocations: 418 bytes)... but random
# After reusing _rand_vec3:
#    92.865 ns (2 allocations: 60 bytes)
# Various speed-ups (don't use Vec3, etc.): 
#    52.427 ns (0 allocations: 0 bytes)
# Float64: 34.549 ns (0 allocations: 0 bytes)
# Float32: 36.274 ns (0 allocations: 0 bytes)
# rand() using Xoroshiro128Plus _rng w/ Float64:
#   19.937 ns (0 allocations: 0 bytes)
random_vec3_on_sphere(Float32)

@inline function random_vec2_in_disk(::Type{T}) where T # equiv to random_in_unit_disk()
	while true
		p = random_vec2(T(-1), T(1))
		if p⋅p <= 1
			return p
		end
	end
end

# Keep best timing of 5!
# Float32: 14.391 ns (0 allocations: 0 bytes)
# Float64: 14.392 ns (0 allocations: 0 bytes)
# rand() using MersenneTwister _rng w/ Float64:
#   7.925 ns (0 allocations: 0 bytes)
# rand() using Xoroshiro128Plus _rng w/ Float64:
#   7.574 ns (0 allocations: 0 bytes)
random_vec2_in_disk(Float64)

""" Temporary function to shoot rays through each pixel. Later replaced by `render`
	
	Args:
		scene: a function that takes a ray, returns the color of any object it hit
"""
function main(nx::Int, ny::Int, scene, ::Type{T}) where T
	lower_left_corner = SA[-2,-1,-1]
	horizontal = SA[4,0,0]
	vertical = SA[T(0),T(2),T(0)]
	origin = SA[T(0),T(0),T(0)]
	
	img = zeros(RGB{T}, ny, nx)
	Threads.@threads for j in 1:nx
		@inbounds for i in 1:ny # Julia is column-major, i.e. iterate 1 column at a time
			u = T(j/nx)
			v = T((ny-i)/ny) # Y-up!
			ray = Ray(origin, normalize(lower_left_corner + u*horizontal + v*vertical))
			#r = x/nx
			#g = y/ny
			#b = 0.2
			img[i,j] = rgb(scene(ray))
		end
	end
	img
end

# Before optimizations:
#   28.630 ms (520010 allocations: 13.05 MiB)
# After optimizations (skycolor, rand, etc.)
#   10.358 ms (220010 allocations: 5.42 MiB) # at this stage, rgb() is most likely the culprit..
# After replacing Color,Vec3 by @SVector:
#    6.539 ms (80002 allocations: 1.75 MiB)
# Lots more optimizations, including @inline of low-level functions:
#  161.546 μs (     2 allocations: 234.45 KiB)
# With Vec3{T}...
# Float32: 174.560 μs (2 allocations: 234.45 KiB)
# Float64: 161.516 μs (2 allocations: 468.83 KiB)
# Above was all using 1 single thread. With 16 threads:
#   52.218 μs (83 allocations: 241.75 KiB)
# With @inbounds:
#   49.243 μs (83 allocations: 241.75 KiB)
main(200, 100, skycolor, Float32) 

#md"# Chapter 5: Add a sphere"

# @fastmath speeds up a lot!
@inline @fastmath function hit_sphere1(center::Vec3{T}, radius::T, r::Ray{T}) where T
	oc = r.origin - center
	#a = r.dir ⋅ r.dir # unnecessary since `r.dir` is normalized
	a = 1
	b = 2oc ⋅ r.dir
	c = (oc ⋅ oc) - radius*radius
	discriminant = b*b - 4a*c
	discriminant > 0
end

@inline function sphere_scene1(r::Ray{T}) where T
	if hit_sphere1(SA[T(0), T(0), T(-1)], T(0.5), r) # sphere of radius 0.5 centered at z=-1
		return SA[T(1),T(0),T(0)] # red
	else
		skycolor(r)
	end
end

# Float32: 149.613 μs (2 allocations: 234.45 KiB)
# Float64: 173.818 μs (2 allocations: 468.83 KiB)
# w/ @fastmath in hit_sphere1:
#    63.751 μs (83 allocations: 476.14 KiB)
# Eliminate unnecessary r.dir ⋅ r.dir: 
#    61.416 μs (83 allocations: 476.14 KiB)
main(200, 100, sphere_scene1, Float64) 

#md"# Chapter 6: Surface normals and multiple objects"

@inline function hit_sphere2(center::Vec3{T}, radius::T, r::Ray{T}) where T
	oc = r.origin - center
	#a = r.dir ⋅ r.dir # unnecessary since `r.dir` is normalized
	a = 1
	b = 2oc ⋅ r.dir
	c = (oc ⋅ oc) - radius^2
	discriminant = b*b - 4a*c
	if discriminant < 0
		return -1
	else
		return (-b - sqrt(discriminant)) / 2a
	end
end

@inline function sphere_scene2(r::Ray{T}) where T
	sphere_center = SA[T(0),T(0),T(-1)]
	t = hit_sphere2(sphere_center, T(0.5), r) # sphere of radius 0.5 centered at z=-1
	if t > T(0)
		n⃗ = normalize(point(r, t) - sphere_center) # normal vector. typed n\vec
		return T(0.5)*n⃗ + SA[T(0.5),T(0.5),T(0.5)] # remap normal to rgb
	else
		skycolor(r)
	end
end

# claforte: this got significantly worse after Vec3{T} was integrated...
# WAS: 
# Float32: 262.286 μs (2 allocations: 234.45 KiB)
# NOW:
# Float32: 290.470 μs (2 allocations: 234.45 KiB)
# Float64: 351.164 μs (2 allocations: 468.83 KiB)
# Above was all using 1 single thread. With 16 threads:
#   64.612 μs (83 allocations: 241.75 KiB)
# Eliminate unnecessary r.dir ⋅ r.dir: 
#   61.586 μs (82 allocations: 241.72 KiB)
#print("@btime main(200,100,sphere_scene2, Float32):")
main(200,100,sphere_scene2, Float32)

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

# md"""The geometry defines an `outside normal`. A HitRecord stores the `local normal`.
# ![Surface normal](https://raytracing.github.io/images/fig-1.06-normal-sides.jpg)
# """

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

#"Diffuse material"
struct Lambertian{T} <: Material{T}
	albedo::Vec3{T}
end

"""Compute reflection vector for v (pointing to surface) and normal n⃗.

	See [diagram](https://raytracing.github.io/books/RayTracingInOneWeekend.html#metal/mirroredlightreflection)"""
#reflect(v::SVector{3,Float32}, n⃗::SVector{3,Float32}) = normalize(v - (2v⋅n⃗)*n⃗) # claforte: normalize not needed?
@inline @fastmath reflect(v::Vec3{T}, n⃗::Vec3{T}) where T = v - (2v⋅n⃗)*n⃗

@assert reflect(SA[0.6,-0.8,0.0], SA[0.0,1.0,0.0]) == SA[0.6,0.8,0.0] # diagram's example

"""Create a scattered ray emitted by `mat` from incident Ray `r`. 

	Args:
		rec: the HitRecord of the surface from which to scatter the ray.

	Return `nothing`` if it's fully absorbed. """
@inline @fastmath function scatter(mat::Lambertian{T}, r::Ray{T}, rec::HitRecord{T})::Scatter{T} where T
	scatter_dir = rec.n⃗ + random_vec3_on_sphere(T)
	if near_zero(scatter_dir) # Catch degenerate scatter direction
		scatter_dir = rec.n⃗ 
	else
		scatter_dir = normalize(scatter_dir)
	end
	scattered_r = Ray{T}(rec.p, scatter_dir)
	attenuation = mat.albedo
	return Scatter(scattered_r, attenuation)
end

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

#"""Find closest hit between `Ray r` and a list of Hittable objects `h`, within distance `tmin` < `tmax`"""
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

#md"# Metal material"

struct Metal{T} <: Material{T}
	albedo::Vec3{T}
	fuzz::T # how big the sphere used to generate fuzzy reflection rays. 0=none
	@inline Metal(a::Vec3{T}, f::T=0.0) where T = new{T}(a,f)
end
@inline @fastmath function scatter(mat::Metal{T}, r_in::Ray{T}, rec::HitRecord)::Scatter{T} where T
	reflected = normalize(reflect(r_in.dir, rec.n⃗) + mat.fuzz*random_vec3_on_sphere(T))
	Scatter(Ray(rec.p, reflected), mat.albedo)
end


#md"# Scenes"

#"Scene with 2 Lambertian spheres"
function scene_2_spheres(; elem_type::Type{T}) where T
	spheres = Sphere[]
	
	# small center sphere
	push!(spheres, Sphere((SA{T}[0,0,-1]), T(0.5), Lambertian(SA{T}[0.7,0.3,0.3])))
	
	# ground sphere
	push!(spheres, Sphere((SA{T}[0,-100.5,-1]), T(100), Lambertian(SA{T}[0.8,0.8,0.0])))
	HittableList(spheres)
end

#"""Scene with 2 Lambertian, 2 Metal spheres.
#
#	See https://raytracing.github.io/images/img-1.11-metal-shiny.png"""
function scene_4_spheres(; elem_type::Type{T}) where T
	scene = scene_2_spheres(; elem_type=elem_type)

	# left and right Metal spheres
	push!(scene, Sphere((SA{T}[-1,0,-1]), T(0.5), Metal((SA{T}[0.8,0.8,0.8]), T(0.3)))) 
	push!(scene, Sphere((SA{T}[ 1,0,-1]), T(0.5), Metal((SA{T}[0.8,0.6,0.2]), T(0.8))))
	return scene
end

#md"""# Camera

# Adapted from C++'s sections 7.2, 11.1 """
struct Camera{T <: AbstractFloat}
	origin::Vec3{T}
	lower_left_corner::Vec3{T}
	horizontal::Vec3{T}
	vertical::Vec3{T}
	u::Vec3{T}
	v::Vec3{T}
	w::Vec3{T}
	lens_radius::T
end

"""
	Args:
		vfov: vertical field-of-view in degrees
		aspect_ratio: horizontal/vertical ratio of pixels
      aperture: if 0 - no depth-of-field
"""
function default_camera(lookfrom::Vec3{T}=(SA{T}[0,0,0]), 
						lookat::Vec3{T}=(SA{T}[0,0,-1]), 
						vup::Vec3{T}=(SA{T}[0,1,0]), 
						vfov::T=T(90), aspect_ratio::T=T(16/9),
						aperture::T=T(0), focus_dist::T=T(1)) where T
	viewport_height = T(2) * tand(vfov/T(2))
	viewport_width = aspect_ratio * viewport_height
	
	w = normalize(lookfrom - lookat)
	u = normalize(vup × w)
	v = w × u
	
	origin = lookfrom
	horizontal = focus_dist * viewport_width * u
	vertical = focus_dist * viewport_height * v
	lower_left_corner = origin - horizontal/T(2) - vertical/T(2) - focus_dist*w
	lens_radius = aperture/T(2)
	Camera{T}(origin, lower_left_corner, horizontal, vertical, u, v, w, lens_radius)
end

default_camera(SA[0f0,0f0,0f0]) # Float32 camera

default_camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist; elem_type::Type{T}) where T =
	default_camera(Vec3{T}(lookfrom), Vec3{T}(lookat), Vec3{T}(vup), 
		T(vfov), T(aspect_ratio), T(aperture), T(focus_dist)
	)

#md"# Render

@inline @fastmath function get_ray(c::Camera{T}, s::T, t::T) where T
	rd = SVector{2,T}(c.lens_radius * random_vec2_in_disk(T))
	offset = c.u * rd.x + c.v * rd.y #offset = c.u * rd.x + c.v * rd.y
    Ray(c.origin + offset, normalize(c.lower_left_corner + s*c.horizontal +
									 t*c.vertical - c.origin - offset))
end

get_ray(default_camera(SA[0f0,0f0,0f0]), 0.0f0, 0.0f0)
#@btime get_ray(default_camera(), 0.0f0, 0.0f0)

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

# Float32/Float64
ELEM_TYPE = Float64

t_default_cam = default_camera(SA{ELEM_TYPE}[0,0,0])

# After some optimization:
#  46.506 ms (917106 allocations: 16.33 MiB)
# Using convert(Float32, ...) instead of MyFloat(...):
#  14.137 ms (118583 allocations: 4.14 MiB)
# Don't specify return value of Option{HitRecord} in hit()
#  24.842 ms (527738 allocations: 16.63 MiB)
# Don't initialize unnecessary elements in HitRecord(): 
#  14.862 ms (118745 allocations: 4.15 MiB)  (but computer was busy...)
# Replace MyFloat by Float32:
#  11.792 ms ( 61551 allocations: 2.88 MiB)
# Remove ::HitRecord return value in remaining hit() method:
#  11.545 ms ( 61654 allocations: 2.88 MiB)
# with mutable HitRecord
#  11.183 ms ( 61678 allocations: 2.88 MiB) (insignificant?)
# @inline tons of stuff. Note: render() uses `for i in 1:image_height, j in 1:image_width`,
#    i.e. iterating 1 row at time!
#   8.129 ms ( 61660 allocations: 2.88 MiB)
# Using in render(): `for j in 1:image_width, i in 1:image_height # iterate over each column`
#  10.489 ms ( 61722 allocations: 2.88 MiB) (consistently slower!)
# ... sticking with `for i in 1:image_height, j in 1:image_width # iterate over each row` for now...
# Using `get_ray(cam, u+δu, v+δv)` (fixes minor bug, extract constants outside inner loop):
# ... performance appears equivalent, maybe a tiny bit faster on avg (1%?)
# Re-measured:
#   8.077 ms (61610 allocations: 2.88 MiB)
# After parameterized Vec3{T}:
# Float64: 8.344 ms (61584 allocations: 4.82 MiB)
# Float32: 8.750 ms (123425 allocations: 3.83 MiB)
# rand() using MersenneTwister _rng w/ Float64:
#   6.967 ms (61600 allocations: 4.82 MiB)
# rand() using Xoroshiro128Plus _rng w/ Float64:
#   6.536 ms (61441 allocations: 4.81 MiB)
# Above was all using 1 single thread. With 16 threads:
#   4.414 ms (61673 allocations: 4.82 MiB)
# Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
# Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
#   4.465 ms (65680 allocations: 5.13 MiB)
# Lots of optimizations... ending with make HitRecord non-mutable:
#   2.225 ms (445188 allocations: 34.08 MiB)
# Using non-mutable HitRecord, Union{HitRecord,Missing}, ismissing():
#   976.365 μs (65574 allocations: 5.12 MiB)
# Using @paulmelis' style of hit(): @inbounds for i in eachindex(hittables) and Union{HitRecord, Nothing}
#   951.447 μs (65574 allocations: 5.12 MiB)
render(scene_2_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 16) # 16 samples

# Iterate over each column: 614.820 μs
# Iterate over each row: 500.334 μs
# With Rand(Float32) everywhere:
#   489.745 μs (3758 allocations: 237.02 KiB)
# With parameterized Vec3{T}:
# Float64: 530.957 μs (3760 allocations: 414.88 KiB)
# rand() using MersenneTwister _rng w/ Float64:
#   473.672 μs (3748 allocations: 413.94 KiB)
# rand() using Xoroshiro128Plus _rng w/ Float64:
#   444.399 μs (3737 allocations: 413.08 KiB)
# Above was all using 1 single thread. With 16 threads:
#   300.438 μs (3829 allocations: 420.86 KiB)
# Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
# Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
#   314.094 μs (4009 allocations: 434.97 KiB)
# Lots of optimizations... ending with make HitRecord non-mutable:
#   136.388 μs (28306 allocations: 2.28 MiB)
# Using non-mutable HitRecord, Union{HitRecordMissing}, ismissing():
#   102.764 μs (4314 allocations: 459.41 KiB)
# Using @paulmelis' style of hit(): @inbounds for i in eachindex(hittables) and Union{HitRecord, Nothing}
#   101.161 μs (4314 allocations: 459.41 KiB)
render(scene_2_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 1) # 1 sample

#render(scene_4_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 16)

#md"""# Dielectrics

# from Section 10.2 Snell's Law:
# ![Ray refraction](https://raytracing.github.io/images/fig-1.13-refraction.jpg)

# Refracted angle `sinθ′ = (η/η′)⋅sinθ`, where η (\eta) are the refractive indices.

# Split the parts of the ray into `R′=R′⊥+R′∥` (perpendicular and parallel to n⃗′)."""

# """
# 	Args:
# 		refraction_ratio: incident refraction index divided by refraction index of 
# 			hit surface. i.e. η/η′ in the figure above"""
@inline @fastmath function refract(dir::Vec3{T}, n⃗::Vec3{T}, refraction_ratio::T) where T
	cosθ = min(-dir ⋅ n⃗, one(T))
	r_out_perp = refraction_ratio * (dir + cosθ*n⃗)
	r_out_parallel = -√(abs(one(T)-squared_length(r_out_perp))) * n⃗
	normalize(r_out_perp + r_out_parallel)
end

# unchanged angle
@assert refract((@SVector[0.6,-0.8,0]), (@SVector[0.0,1.0,0.0]), 1.0) == @SVector[0.6,-0.8,0.0] 

# wider angle
t_refract_widerθ = refract(@SVector[0.6,-0.8,0.0], @SVector[0.0,1.0,0.0], 2.0)
@assert isapprox(t_refract_widerθ, @SVector[0.87519,-0.483779,0.0]; atol=1e-3)

# narrower angle
t_refract_narrowerθ = refract(@SVector[0.6,-0.8,0.0], @SVector[0.0,1.0,0.0], 0.5)
@assert isapprox(t_refract_narrowerθ, @SVector[0.3,-0.953939,0.0]; atol=1e-3)

struct Dielectric{T} <: Material{T}
	ir::T # index of refraction, i.e. η.
end

@inline @fastmath function reflectance(cosθ, refraction_ratio)
	# Use Schlick's approximation for reflectance.
	# claforte: may be buggy? I'm getting black pixels in the Hollow Glass Sphere...
	r0 = (1-refraction_ratio) / (1+refraction_ratio)
	r0 = r0^2
	r0 + (1-r0)*((1-cosθ)^5)
end

@inline @fastmath function scatter(mat::Dielectric{T}, r_in::Ray{T}, rec::HitRecord{T}) where T
	attenuation = SA{T}[1,1,1]
	refraction_ratio = rec.front_face ? (one(T)/mat.ir) : mat.ir # i.e. ηᵢ/ηₜ
	cosθ = min(-r_in.dir⋅rec.n⃗, one(T))
	sinθ = √(one(T) - cosθ^2)
	cannot_refract = refraction_ratio * sinθ > one(T)
	if cannot_refract || reflectance(cosθ, refraction_ratio) > trand(T)
		dir = reflect(r_in.dir, rec.n⃗)
	else
		dir = refract(r_in.dir, rec.n⃗, refraction_ratio)
	end
	Scatter(Ray{T}(rec.p, dir), attenuation) # TODO: rename reflected -> !absorbed?
end

#"From C++: Image 15: Glass sphere that sometimes refracts"
@inline function scene_diel_spheres(left_radius=0.5; elem_type::Type{T}) where T # dielectric spheres
	spheres = Sphere[]
	
	# small center sphere
	push!(spheres, Sphere((SA{T}[0,0,-1]), T(0.5), Lambertian(SA{T}[0.1,0.2,0.5])))
	
	# ground sphere (planet?)
	push!(spheres, Sphere((SA{T}[0,-100.5,-1]), T(100), Lambertian(SA{T}[0.8,0.8,0.0])))
	
	# # left and right spheres.
	# # Use a negative radius on the left sphere to create a "thin bubble" 
	push!(spheres, Sphere((SA{T}[-1,0,-1]), T(left_radius), Dielectric(T(1.5)))) 
	push!(spheres, Sphere((SA{T}[1,0,-1]), T(0.5), Metal((SA{T}[0.8,0.6,0.2]), T(0))))
	HittableList(spheres)
end

#render(scene_diel_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 16)
#render(scene_diel_spheres(), default_camera(), 320, 32)

# Hollow Glass sphere using a negative radius
#render(scene_diel_spheres(-0.5; elem_type=ELEM_TYPE), t_default_cam, 96, 16)

#render(scene_diel_spheres(; elem_type=ELEM_TYPE), default_camera((SA{ELEM_TYPE}[-2,2,1]), (SA{ELEM_TYPE}[0,0,-1]),
#																 (SA{ELEM_TYPE}[0,1,0]), ELEM_TYPE(20)), 96, 16)


#md"# Positioning camera"

function scene_blue_red_spheres(; elem_type::Type{T}) where T # dielectric spheres
	spheres = Sphere[]
	R = cos(pi/4)
	push!(spheres, Sphere((SA{T}[-R,0,-1]), R, Lambertian(SA{T}[0,0,1]))) 
	push!(spheres, Sphere((SA{T}[ R,0,-1]), R, Lambertian(SA{T}[1,0,0]))) 
	HittableList(spheres)
end

#render(scene_blue_red_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 16)

#md"# Random spheres"

function scene_random_spheres(; elem_type::Type{T}) where T
	spheres = Sphere[]

	# ground 
	push!(spheres, Sphere((SA{T}[0,-1000,-1]), T(1000), 
						  Lambertian(SA{T}[0.5,0.5,0.5])))

	for a in -11:10, b in -11:10
		choose_mat = trand(T)
		center = SA[a + T(0.9)*trand(T), T(0.2), b + T(0.9)*trand(T)]

		# skip spheres too close?
		if norm(center - SA{T}[4,0.2,0]) < T(0.9) continue end 
			
		if choose_mat < T(0.8)
			# diffuse
			albedo = @SVector[trand(T) for i ∈ 1:3] .* @SVector[trand(T) for i ∈ 1:3]
			push!(spheres, Sphere(center, T(0.2), Lambertian(albedo)))
		elseif choose_mat < T(0.95)
			# metal
			albedo = @SVector[random_between(T(0.5),T(1.0)) for i ∈ 1:3]
			fuzz = random_between(T(0.0), T(5.0))
			push!(spheres, Sphere(center, T(0.2), Metal(albedo, fuzz)))
		else
			# glass
			push!(spheres, Sphere(center, T(0.2), Dielectric(T(1.5))))
		end
	end

	push!(spheres, Sphere((SA{T}[0,1,0]), T(1), Dielectric(T(1.5))))
	push!(spheres, Sphere((SA{T}[-4,1,0]), T(1), 
						  Lambertian(SA{T}[0.4,0.2,0.1])))
	push!(spheres, Sphere((SA{T}[4,1,0]), T(1), 
						  Metal((SA{T}[0.7,0.6,0.5]), T(0))))
	HittableList(spheres)
end

t_cam1 = default_camera([13,2,3], [0,0,0], [0,1,0], 20, 16/9, 0.1, 10.0; elem_type=ELEM_TYPE)

# took ~20s (really rough timing) in REPL, before optimization
# after optimization: 
#   880.997 ms (48801164 allocations: 847.10 MiB)
# after switching to Float32 + reducing allocations using rand_vec3!(): 
#    79.574 ms (  542467 allocations:  14.93 MiB)
# after optimizing skycolor, rand*, probably more stuff I forgot...
#    38.242 ms (  235752 allocations:   6.37 MiB)
# after removing all remaining Color, Vec3, replacing them with @SVector[]...
#    26.790 ms (   91486 allocations: 2.28 MiB)
# Using convert(Float32, ...) instead of MyFloat(...):
#    25.856 ms (   70650 allocations: 1.96 MiB)
# Don't specify return value of Option{HitRecord} in hit()
# Don't initialize unnecessary elements in HitRecord(): 
#    38.961 ms (   70681 allocations: 1.96 MiB) # WORSE, probably because we're writing a lot more to stack...?
# Replace MyFloat by Float32:
#    36.726 ms (13889 allocations: 724.09 KiB)
# @inline lots of stuff:
#    14.690 ms (13652 allocations: 712.98 KiB)
# rand(Float32) to avoid Float64s:
#    14.659 ms (13670 allocations: 713.84 KiB)
# Re-measured:
#    14.069 ms (13677 allocations: 714.22 KiB)
# After parameterized Vec3{T}:
# Float32: 14.422 ms (26376 allocations: 896.52 KiB) # claforte: why are there 2X the allocations as previously?
# Float64: 12.772 ms (12868 allocations: 1.08 MiB) (10% speed-up! Thanks @woclass!)
# rand() using MersenneTwister _rng w/ Float64:
#    14.092 ms (12769 allocations: 1.07 MiB) (SLOWER?!)
# rand() using Xoroshiro128Plus _rng w/ Float64:
#    12.581 ms (12943 allocations: 1.08 MiB)
# Above was all using 1 single thread. With 16 threads:
#     1.789 ms (12926 allocations: 1.08 MiB) (WOW!)
# Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
# Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
#     2.168 ms (13791 allocations: 1.15 MiB)
# Using  bunch of @inbounds, @simd in low-level functions
#     2.076 ms (13861 allocations: 1.15 MiB)
# Lots of optimizations, up to `Using non-mutable HitRecord, Union{HitRecordMissing}, ismissing():`
#     2.042 ms (14825 allocations: 1.23 MiB)
#@btime render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 96, 1)

# took 5020s in Pluto.jl, before optimizations!
# after lots of optimizations, up to switching to Float32 + reducing allocations using rand_vec3!(): 
#   10.862032 seconds (70.09 M allocations: 1.858 GiB, 1.53% gc time)
# after optimizing skycolor, rand*, probably more stuff I forgot...
#    4.926 s (25,694,163 allocations: 660.06 MiB)
# after removing all remaining Color, Vec3, replacing them with @SVector[]...
#    3.541 s (9074843 allocations: 195.22 MiB)
# Using convert(Float32, ...) instead of MyFloat(...):
#    3.222 s (2055106 allocations: 88.53 MiB)
# Don't specify return value of Option{HitRecord} in hit()
# Don't initialize unnecessary elements in HitRecord(): 
#    5.016 s (2056817 allocations: 88.61 MiB) #  WORSE, probably because we're writing a lot more to stack...?
# Replace MyFloat by Float32:
#    5.185 s (1819234 allocations: 83.55 MiB) # Increase of 1% is probably noise
# Remove normalize() in reflect() (that function assumes the inputs are normalized)
#    5.040 s (1832052 allocations: 84.13 MiB)
# @inline lots of stuff:
#    2.110 s (1823044 allocations: 83.72 MiB)
# rand(Float32) to avoid Float64s:
#    2.063 s (1777985 allocations: 81.66 MiB)
# @woclass's "Use alias instead of new struct", i.e. `const HittableList = Vector{Hittable}`:
#    1.934 s (1796954 allocations: 82.53 MiB)
# @woclass's Vec3{T} with T=Float64: (7.8% speed-up!)
#    1.800 s (1711061 allocations: 131.03 MiB) 
# rand() using Xoroshiro128Plus _rng w/ Float64:
#    1.808 s (1690104 allocations: 129.43 MiB) (i.e. rand is not a bottleneck?)
# Above was all using 1 single thread. With 16 threads:
#  265.331 ms (1645402 allocations: 126.02 MiB) (WOW!)
# Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
# Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
#  308.217 ms (1830162 allocations: 140.12 MiB)
# Using @inbounds, @simd in low-level functions:
#  302.952 ms (1892513 allocations: 144.88 MiB)
# Convert Camera and every Material structs to non-mutable:
#  301.042 ms (1849711 allocations: 141.61 MiB)  (i.e. - unchanged)
# Adapt @Christ_Foster's Base.getproperty w/ @inline @inbounds:
#  292.603 ms (1856398 allocations: 142.12 MiB) (ran multiple times, seems like real, 3-5% speed-up)
# Eliminate the off-by-half-a-pixel offset:
#  286.873 ms (1811412 allocations: 138.69 MiB) (ran multiple times, seems like ~2.5% speed-up)
# Fixed, per-thread RNGs with fixed seeds
#  286.575 ms (1884433 allocations: 144.26 MiB) (i.e. maybe a tiny bit faster considering this fixed seed has more allocations?)
# Make HitRecord non-mutable:
#   29.733 s (937962909 allocations: 69.88 GiB) (WTF!)
# Lots of optimizations, up to `Using non-mutable HitRecord, Union{HitRecordMissing}, ismissing():`
#  306.011 ms (1884433 allocations: 144.26 MiB) (Still slower... Hum)
# Using @paulmelis' style of hit(): @inbounds for i in eachindex(hittables) and Union{HitRecord, Nothing}
#  304.877 ms (1884433 allocations: 144.26 MiB)
# Extract the scene creation from the render() call:
#  300.344 ms (1883484 allocations: 144.21 MiB)
print("render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 200, 32):")
reseed!()
_scene_random_spheres = scene_random_spheres(; elem_type=ELEM_TYPE)
@btime render($_scene_random_spheres, $t_cam1, 200, 32) 

# After some optimization, took ~5.6 hours:
#   20171.646846 seconds (94.73 G allocations: 2.496 TiB, 1.06% gc time)
# ... however the image looked weird... too blurry
# After removing all remaining Color, Vec3, replacing them with @SVector[]... took ~3.7 hours:
#   13326.770907 seconds (29.82 G allocations: 714.941 GiB, 0.36% gc time)
# Using convert(Float32, ...) instead of MyFloat(...):
# Don't specify return value of Option{HitRecord} in hit()
# Don't initialize unnecessary elements in HitRecord(): 
# Took ~4.1 hours:
#   14723.339976 seconds (5.45 G allocations: 243.044 GiB, 0.11% gc time) # WORSE, probably because we're writing a lot more to stack...?
# Replace MyFloat by Float32:
# Lots of other optimizations including @inline lots of stuff: 
#    6018.101653 seconds (5.39 G allocations: 241.139 GiB, 0.21% gc time) (1.67 hour)
# @woclass's rand(Float32) to avoid Float64s: (expected to provide 2.2% speed up)
# @woclass's "Use alias instead of new struct", i.e. `const HittableList = Vector{Hittable}`
# @woclass's Vec3{T} with T=Float64: (7.8% speed-up!): 
#    5268.175362 seconds (4.79 G allocations: 357.005 GiB, 0.47% gc time) (1.46 hours)
# Above was all using 1 single thread. With 16 threads: (~20 minutes)
#    1210.363539 seconds (4.94 G allocations: 368.435 GiB, 10.08% gc time)
# Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
# Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
#    1298.522674 seconds (5.43 G allocations: 404.519 GiB, 10.18% gc time)
# Using @inbounds, @simd in low-level functions:
#    1314.510565 seconds (5.53 G allocations: 411.753 GiB, 10.21% gc time) # NOTE: difference due to randomness?
# Adapt @Christ_Foster's Base.getproperty w/ @inline @inbounds: (expect 3-5% speed-up)
# Eliminate the off-by-half-a-pixel offset: (expect ~2.5% speed-up)
# Fixed, per-thread RNGs with fixed seeds (expecting no noticeable change in speed)
#  Using 16 threads: (21m22s)
#    1282.437499 seconds (5.53 G allocations: 411.742 GiB, 10.08% gc time) (i.e. 2.5% speed-up... currently GC- and memory-bound?)
#  Using 14 threads: (21m45s)
#    1305.767627 seconds (5.53 G allocations: 411.741 GiB, 9.97% gc time)
#print("@time render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 1920, 1000):")
#@time render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 1920, 1000)


t_cam2 = default_camera([3,3,2], [0,0,-1], [0,1,0], 20, 16/9, 2.0, norm([3,3,2]-[0,0,-1]); 
						elem_type=ELEM_TYPE)

# Before optimization:
#  5.993 s  (193097930 allocations: 11.92 GiB)
# after disabling: `Base.getproperty(vec::SVector{3}, sym::Symbol)`
#  1.001 s  ( 17406437 allocations: 425.87 MiB)
# after forcing Ray and point() to use Float64 instead of AbstractFloat:
#  397.905 ms (6269207 allocations: 201.30 MiB)
# after forcing use of Float32 instead of Float64:
#  487.680 ms (7128113 allocations: 196.89 MiB) # More allocations... something is causing them...
# after optimizing rand_vec3!/rand_vec2! to minimize allocations:
#  423.468 ms (6075725 allocations: 158.92 MiB)
# after optimizing skycolor, rand*, probably more stuff I forgot...
#  217.853 ms (2942272 allocations: 74.82 MiB)
# after removing all remaining Color, Vec3, replacing them with @SVector[]...
#   56.778 ms (1009344 allocations: 20.56 MiB)
# Using convert(Float32, ...) instead of MyFloat(...):
#   23.870 ms (210890 allocations: 8.37 MiB)
# Replace MyFloat by Float32:
#   22.390 ms (153918 allocations: 7.11 MiB)
# Various other changes, e.g. remove unnecessary normalize
#   20.241 ms (153792 allocations: 7.10 MiB)
# @inline lots of stuff:
#   18.065 ms (153849 allocations: 7.10 MiB)
# rand(Float32) to avoid Float64s:
#   16.035 ms (153777 allocations: 7.10 MiB)
# After @woclass's Vec3{T} with T=Float64:
#   16.822 ms (153591 allocations: 11.84 MiB)
# rand() using Xoroshiro128Plus _rng w/ Float64:
#   13.469 ms (153487 allocations: 11.83 MiB)
# Above was all using 1 single thread. With 16 threads:
#    6.537 ms (153599 allocations: 11.84 MiB)
# Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
# Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
#    6.766 ms (161000 allocations: 12.40 MiB)
# @inbounds and @simd in low-level functions
#    6.519 ms (160609 allocations: 12.37 MiB)
#render(scene_diel_spheres(; elem_type=ELEM_TYPE), t_cam2, 96, 16)

using Profile
render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 16, 1)
Profile.clear_malloc_data()
render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 17, 13)
