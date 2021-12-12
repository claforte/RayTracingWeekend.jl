# Prototype - copied from pluto_RayTracingWeekend.jl
# Adapted from [Ray Tracing In One Weekend by Peter Shirley](https://raytracing.github.io/books/RayTracingInOneWeekend.html) and 
# [cshenton's Julia implementation](https://github.com/cshenton/RayTracing.jl)"
using Pkg
Pkg.activate(@__DIR__)

"""Next ideas:

- Incorporate ideas from https://discourse.julialang.org/t/why-is-this-raytracer-slow/59176/20
  - @inline to many functions
- save image, e.g. PNG
- The only remaining allocations that appear expensive are of `HitRecord`s (on the stack)
- pre-allocate X paths (ray bundles? not sure what the terminology is) then 
  implement versions of hit, scatter, etc. that operate on an entire tensor at once.
  (i.e. efficiently parallelizable with multithreading, on SIMD or GPU)
  - use FieldVector?
- figure out the incorrect look in refraction of negatively scaled sphere
- use parameterizable structs for Vec but keep the same efficiency
- share with community, ask for feedback
- continue watching MIT course
"""

using BenchmarkTools, Images, InteractiveUtils, LinearAlgebra, StaticArrays
Option{T} = Union{Missing, T}
t_col = @SVector[0.4f0, 0.5f0, 0.1f0] # test color

squared_length(v::SVector{3,Float32}) = v ⋅ v

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
#@btime squared_length($t_col)

near_zero(v::SVector{3,Float32}) = squared_length(v) < 1e-5
#@btime near_zero($t_col) # 1.382 ns (0 allocations: 0 bytes)

# Test images

img = rand(4, 3)

img_rgb = rand(RGB{Float32}, 4, 4)

ex1 = [RGB{Float32}(1,0,0) RGB{Float32}(0,1,0) RGB{Float32}(0,0,1);
       RGB{Float32}(1,1,0) RGB{Float32}(1,1,1) RGB{Float32}(0,0,0)]

ex2 = zeros(RGB{Float32}, 2, 3)
ex2[1,1] = RGB{Float32}(1,0,0)
ex2

# The "Hello World" of graphics
function gradient(nx::Int, ny::Int)
	img = zeros(RGB{Float32}, ny, nx)
	for i in 1:ny, j in 1:nx # Julia is column-major, i.e. iterate 1 column at a time
		x = j; y = (ny-i);
		r = x/nx
		g = y/ny
		b = 0.2
		img[i,j] = RGB{Float32}(r,g,b)
	end
	img
end

gradient(200,100)

rgb(v::SVector{3,Float32}) = RGB{Float32}(v[1], v[2], v[3])
#rgb(v::Vec3) = RGB{MyFloat}(v...)
#@btime rgb($t_col) # 289.914 ns (4 allocations: 80 bytes)

rgb_gamma2(v::SVector{3,Float32}) = RGB{Float32}(sqrt.(v)...)

#@btime rgb_gamma2($t_col) # 454.766 ns (11 allocations: 256 bytes)

struct Ray
	origin::SVector{3, Float32} # Point 
	dir::SVector{3, Float32} # Vec3 # direction (unit vector)
end

# interpolates between blue and white
_origin = @SVector[0.0f0,0.0f0,0.0f0]
_v3_minusY = @SVector[0.0f0,-1.0f0,0.0f0]
_t_ray1 = Ray(_origin, _v3_minusY)
#@btime Ray($_origin, $_v3_minusY) # 6.341 ns (0 allocations: 0 bytes)


# equivalent to C++'s ray.at()
function point(r::Ray, t::Float32)
	r.origin .+ t .* r.dir
end
#@btime point($_t_ray1, 0.5f0) # 1.392 ns (0 allocations: 0 bytes)

#md"# Chapter 4: Rays, simple camera, and background"

function skycolor(ray::Ray)
	white = @SVector[1.0f0,1.0f0,1.0f0]
	skyblue = @SVector[0.5f0,0.7f0,1.0f0]
	t = 0.5f0(ray.dir[2] + 1.0f0)
    (1.0f0-t)*white + t*skyblue
end
#@btime skycolor($_t_ray1) # 1.402 ns (0 allocations: 0 bytes)

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
#@btime rgb(skycolor($t_ray1)) # 291.492 ns (4 allocations: 80 bytes)

# md"""# Random vectors
# C++'s section 8.1"""

random_between(min=0.0f0, max=1.0f0) = rand(Float32)*(max-min) + min # equiv to random_double()
#@btime random_between(50f0, 100f0) # 4.519 ns (0 allocations: 0 bytes)

function random_vec3(min::Float32, max::Float32)
	@SVector[random_between(min, max) for i in 1:3]
end

# Before optimization:
#   352.322 ns (6 allocations: 224 bytes)
# Don't use list comprehension:
#   179.684 ns (5 allocations: 112 bytes)
# Return @SVector[] instead of Vec3():
#    12.557 ns (0 allocations: 0 bytes)
#@btime random_vec3(-1.0f0,1.0f0)

function random_vec2(min::Float32, max::Float32)
	@SVector[random_between(min, max) for i in 1:2]
end
#@btime random_vec2(-1.0f0,1.0f0) # 8.536 ns (0 allocations: 0 bytes)

function random_vec3_in_sphere() # equiv to random_in_unit_sphere()
	while true
		p = random_vec3(-1f0, 1f0)
		if p⋅p <= 1
			return p
		end
	end
end
#@btime random_vec3_in_sphere() # 46.587 ns (0 allocations: 0 bytes)

squared_length(random_vec3_in_sphere())

"Random unit vector. Equivalent to C++'s `unit_vector(random_in_unit_sphere())`"
random_vec3_on_sphere() = normalize(random_vec3_in_sphere())
#@btime random_vec3_on_sphere()

# Before optim:
#   517.538 ns (12 allocations: 418 bytes)... but random
# After reusing _rand_vec3:
#    92.865 ns (2 allocations: 60 bytes)
# Various speed-ups (don't use Vec3, etc.): 
#    52.427 ns (0 allocations: 0 bytes)
random_vec3_on_sphere()
norm(random_vec3_on_sphere())

function random_vec2_in_disk() # equiv to random_in_unit_disk()
	while true
		p = random_vec2(-1f0, 1f0)
		if p⋅p <= 1
			return p
		end
	end
end
#@btime random_vec2_in_disk() # 16.725 ns (0 allocations: 0 bytes)

""" Temporary function to shoot rays through each pixel. Later replaced by `render`
	
	Args:
		scene: a function that takes a ray, returns the color of any object it hit
"""
function main(nx::Int, ny::Int, scene)
	lower_left_corner = @SVector[-2,-1,-1]
	horizontal = @SVector[4,0,0]
	vertical = @SVector[0f0,2f0,0f0]
	origin = @SVector[0,0,0]
	
	img = zeros(RGB{Float32}, ny, nx)
	for i in 1:ny, j in 1:nx # Julia is column-major, i.e. iterate 1 column at a time
		u = j/nx
		v = (ny-i)/ny # Y-up!
		ray = Ray(origin, normalize(lower_left_corner + u*horizontal + v*vertical))
		#r = x/nx
		#g = y/ny
		#b = 0.2
		img[i,j] = rgb(scene(ray))
	end
	img
end

# Before optimizations:
#   28.630 ms (520010 allocations: 13.05 MiB)
# After optimizations (skycolor, rand, etc.)
#   10.358 ms (220010 allocations: 5.42 MiB) # at this stage, rgb() is most likely the culprit..
# After replacing Color,Vec3 by @SVector:
#    6.539 ms (80002 allocations: 1.75 MiB)
main(200,100, skycolor) 

#md"# Chapter 5: Add a sphere"

function hit_sphere1(center::SVector{3,Float32}, radius::Float32, r::Ray)
	oc = r.origin - center
	a = r.dir ⋅ r.dir
	b = 2oc ⋅ r.dir
	c = (oc ⋅ oc) - radius*radius
	discriminant = b*b - 4a*c
	discriminant > 0
end

function sphere_scene1(r::Ray)
	if hit_sphere1(@SVector[0f0,0f0,-1f0], 0.5f0, r) # sphere of radius 0.5 centered at z=-1
		return @SVector[1f0,0f0,0f0] # red
	else
		skycolor(r)
	end
end

main(200,100,sphere_scene1) # 10.310 ms (220010 allocations: 5.42 MiB)

#md"# Chapter 6: Surface normals and multiple objects"

function hit_sphere2(center::SVector{3,Float32}, radius::Float32, r::Ray)
	oc = r.origin - center
	a = r.dir ⋅ r.dir
	b = 2oc ⋅ r.dir
	c = (oc ⋅ oc) - radius*radius
	discriminant = b*b - 4a*c
	if discriminant < 0
		return -1
	else
		return (-b - sqrt(discriminant)) / 2a
	end
end

function sphere_scene2(r::Ray)
	sphere_center = SVector{3,Float32}(0f0,0f0,-1f0)
	t = hit_sphere2(sphere_center, 0.5f0, r) # sphere of radius 0.5 centered at z=-1
	if t > 0f0
		n⃗ = normalize(point(r, t) - sphere_center) # normal vector. typed n\vec
		return 0.5f0n⃗ + SVector{3,Float32}(0.5f0,0.5f0,0.5f0) # remap normal to rgb
	else
		skycolor(r)
	end
end

main(200,100,sphere_scene2)  # 6.620 ms (80002 allocations: 1.75 MiB)

"An object that can be hit by Ray"
abstract type Hittable end

"""Materials tell us how rays interact with a surface"""
abstract type Material end

struct NoMaterial <: Material end

const _no_material = NoMaterial()
const _y_up = @SVector[0f0,1f0,0f0]

"Record a hit between a ray and an object's surface"
mutable struct HitRecord
	t::Float32 # vector from the ray's origin to the intersection with a surface. 
	
	# If t==Inf32, there was no hit, and all following values are undefined!
	#
	p::SVector{3,Float32} # point of the intersection between an object's surface and a ray
	n⃗::SVector{3,Float32} # surface's outward normal vector, points towards outside of object?
	
	# If true, our ray hit from outside to the front of the surface. 
	# If false, the ray hit from within.
	front_face::Bool
	mat::Material

	HitRecord() = new(Inf32) # no hit!
	HitRecord(t,p,n⃗,front_face,mat) = new(t,p,n⃗,front_face,mat)
end

struct Sphere <: Hittable
	center::SVector{3,Float32}
	radius::Float32
	mat::Material
end

# md"""The geometry defines an `outside normal`. A HitRecord stores the `local normal`.
# ![Surface normal](https://raytracing.github.io/images/fig-1.06-normal-sides.jpg)
# """

"""Equivalent to `hit_record.set_face_normal()`"""
function ray_to_HitRecord(t, p, outward_n⃗, r_dir::SVector{3,Float32}, mat::Material)
	front_face = r_dir ⋅ outward_n⃗ < 0
	n⃗ = front_face ? outward_n⃗ : -outward_n⃗
	rec = HitRecord(t,p,n⃗,front_face,mat)
end

struct Scatter
	r::Ray
	attenuation::SVector{3,Float32}
	
	# claforte: TODO: rename to "absorbed?", i.e. not reflected/refracted?
	reflected::Bool # whether the scattered ray was reflected, or fully absorbed
	Scatter(r,a,reflected=true) = new(r,a,reflected)
end

#"Diffuse material"
mutable struct Lambertian<:Material
	albedo::SVector{3,Float32}
end

"""Compute reflection vector for v (pointing to surface) and normal n⃗.

	See [diagram](https://raytracing.github.io/books/RayTracingInOneWeekend.html#metal/mirroredlightreflection)"""
#reflect(v::SVector{3,Float32}, n⃗::SVector{3,Float32}) = normalize(v - (2v⋅n⃗)*n⃗) # claforte: normalize not needed?
reflect(v::SVector{3,Float32}, n⃗::SVector{3,Float32}) = v - (2v⋅n⃗)*n⃗

@assert reflect(@SVector[0.6f0,-0.8f0,0f0], @SVector[0f0,1f0,0f0]) == @SVector[0.6f0,0.8f0,0f00] # diagram's example

"""Create a scattered ray emitted by `mat` from incident Ray `r`. 

	Args:
		rec: the HitRecord of the surface from which to scatter the ray.

	Return missing if it's fully absorbed. """
function scatter(mat::Lambertian, r::Ray, rec::HitRecord)::Scatter
	scatter_dir = rec.n⃗ + random_vec3_on_sphere()
	if near_zero(scatter_dir) # Catch degenerate scatter direction
		scatter_dir = rec.n⃗ 
	else
		scatter_dir = normalize(scatter_dir)
	end
	scattered_r = Ray(rec.p, scatter_dir)
	attenuation = mat.albedo
	return Scatter(scattered_r, attenuation)
end

const _no_hit = HitRecord()

function hit(s::Sphere, r::Ray, tmin::Float32, tmax::Float32)#::Option{HitRecord} # the return type seems to cause excessive allocations
    oc = r.origin - s.center
    a = 1 #r.dir ⋅ r.dir # normalized vector - always 1
    half_b = oc ⋅ r.dir
    c = oc⋅oc - s.radius^2
    discriminant = half_b^2 - a*c
	if discriminant < 0 return _no_hit end
	sqrtd = √discriminant
	
	# Find the nearest root that lies in the acceptable range
	root = (-half_b - sqrtd) / a	
	if root < tmin || tmax < root
		root = (-half_b + sqrtd) / a
		if root < tmin || tmax < root
			return _no_hit
		end
	end
		
	t = root
	p = point(r, t)
	n⃗ = (p - s.center) / s.radius
	return ray_to_HitRecord(t, p, n⃗, r.dir, s.mat)
end

struct HittableList <: Hittable
    list::Vector{Hittable}
end

#"""Find closest hit between `Ray r` and a list of Hittable objects `h`, within distance `tmin` < `tmax`"""
function hit(hittables::HittableList, r::Ray, tmin::Float32, tmax::Float32)# ::HitRecord
    closest = tmax # closest t so far
    rec = _no_hit
    for h in hittables.list
        temprec = hit(h, r, tmin, closest)
        if temprec !== _no_hit
            rec = temprec
            closest = rec.t # i.e. ignore any further hit > this one's.
        end
    end
    rec
end

color_vec3_in_rgb(v::SVector{3,Float32}) = 0.5normalize(v) + @SVector[0.5f,0.5f,0.5f]

#md"# Metal material"

mutable struct Metal<:Material
	albedo::SVector{3,Float32}
	fuzz::Float32 # how big the sphere used to generate fuzzy reflection rays. 0=none
	Metal(a,f=0.0) = new(a,f)
end

function scatter(mat::Metal, r_in::Ray, rec::HitRecord)::Scatter
	reflected = normalize(reflect(r_in.dir, rec.n⃗) + mat.fuzz*random_vec3_on_sphere())
	Scatter(Ray(rec.p, reflected), mat.albedo)
end


#md"# Scenes"

#"Scene with 2 Lambertian spheres"
function scene_2_spheres()::HittableList
	spheres = Sphere[]
	
	# small center sphere
	push!(spheres, Sphere(@SVector[0f0,0f0,-1f0], 0.5, Lambertian(@SVector[0.7f0,0.3f0,0.3f0])))
	
	# ground sphere (planet?)
	push!(spheres, Sphere(@SVector[0f0,-100.5f0,-1f0], 100, Lambertian(@SVector[0.8f0,0.8f0,0.0f0])))
	HittableList(spheres)
end

#"""Scene with 2 Lambertian, 2 Metal spheres.
#
#	See https://raytracing.github.io/images/img-1.11-metal-shiny.png"""
function scene_4_spheres()::HittableList
	scene = scene_2_spheres()

	# left and right Metal spheres
	push!(scene.list, Sphere(@SVector[-1f0,0f0,-1f0], 0.5f0, Metal(@SVector[0.8f0,0.8f0,0.8f0], 0.3f0))) 
	push!(scene.list, Sphere(@SVector[1f0,0f0,-1f0], 0.5f0, Metal(@SVector[0.8f0,0.6f0,0.2f0], 0.8f0)))
	return scene
end

#md"""# Camera

# Adapted from C++'s sections 7.2, 11.1 """
mutable struct Camera
	origin::SVector{3,Float32}
	lower_left_corner::SVector{3,Float32}
	horizontal::SVector{3,Float32}
	vertical::SVector{3,Float32}
	u::SVector{3,Float32}
	v::SVector{3,Float32}
	w::SVector{3,Float32}
	lens_radius::Float32
end

"""
	Args:
		vfov: vertical field-of-view in degrees
		aspect_ratio: horizontal/vertical ratio of pixels
      aperture: if 0 - no depth-of-field
"""
function default_camera(lookfrom::SVector{3,Float32}=@SVector[0f0,0f0,0f0], lookat::SVector{3,Float32}=@SVector[0f0,0f0,-1f0], 
						vup::SVector{3,Float32}=@SVector[0f0,1f0,0f0], vfov=90.0f0, aspect_ratio=16.0f0/9.0f0,
						aperture=0.0f0, focus_dist=1.0f0)
	viewport_height = 2.0f0 * tand(vfov/2f0)
	viewport_width = aspect_ratio * viewport_height
	
	w = normalize(lookfrom - lookat)
	u = normalize(vup × w)
	v = w × u
	
	origin = lookfrom
	horizontal = focus_dist * viewport_width * u
	vertical = focus_dist * viewport_height * v
	lower_left_corner = origin - horizontal/2f0 - vertical/2f0 - focus_dist*w
	lens_radius = aperture/2f0
	Camera(origin, lower_left_corner, horizontal, vertical, u, v, w, lens_radius)
end

default_camera()

#md"# Render

function get_ray(c::Camera, s::Float32, t::Float32)
	rd = SVector{2,Float32}(c.lens_radius * random_vec2_in_disk())
	offset = c.u * rd[1] + c.v * rd[2] #offset = c.u * rd.x + c.v * rd.y
    Ray(c.origin + offset, normalize(c.lower_left_corner + s*c.horizontal +
									 t*c.vertical - c.origin - offset))
end

#@btime get_ray(default_camera(), 0.0f0, 0.0f0)

"""Compute color for a ray, recursively

	Args:
		depth: how many more levels of recursive ray bounces can we still compute?"""
function ray_color(r::Ray, world::HittableList, depth=4)
    if depth <= 0
		return @SVector[0f0,0f0,0f0]
	end
		
	rec = hit(world, r, 1f-4, Inf32)
    if rec !== _no_hit
		# For debugging, represent vectors as RGB:
		# return color_vec3_in_rgb(rec.p) # show the normalized hit point
		# return color_vec3_in_rgb(rec.n⃗) # show the normal in RGB
		# return color_vec3_in_rgb(rec.p + rec.n⃗)
		# return color_vec3_in_rgb(random_vec3_in_sphere())
		#return color_vec3_in_rgb(rec.n⃗ + random_vec3_in_sphere())

        s = scatter(rec.mat, r, rec)
		if s.reflected
			return s.attenuation .* ray_color(s.r, world, depth-1)
		else
			return @SVector[0f0,0f0,0f0]
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
function render(scene::HittableList, cam::Camera, image_width=400,
				n_samples=1)
	# Image
	aspect_ratio = 16.0f0/9.0f0 # TODO: use cam.aspect_ratio for consistency
	image_height = convert(Int64, floor(image_width / aspect_ratio))

	# Render
	img = zeros(RGB{Float32}, image_height, image_width)
	# Compared to C++, Julia is:
	# 1. column-major, i.e. iterate 1 column at a time, so invert i,j compared to C++
	# 2. 1-based, so no need to subtract 1 from image_width, etc.
	# 3. The array is Y-down, but `v` is Y-up 
	for i in 1:image_height, j in 1:image_width
		accum_color = @SVector[0f0,0f0,0f0]
		for s in 1:n_samples
			u = convert(Float32, j/image_width)
			v = convert(Float32, (image_height-i)/image_height) # i is Y-down, v is Y-up!
			if s != 1 # 1st sample is always centered, for 1-sample/pixel
				# claforte: I think the C++ version had a bug, the rand offset was
				# between [0,1] instead of centered at 0, e.g. [-0.5, 0.5].
				u += convert(Float32, (rand()-0.5f0) / image_width)
				v += convert(Float32 ,(rand()-0.5f0) / image_height)
			end
			ray = get_ray(cam, u, v)
			accum_color += ray_color(ray, scene)
		end
		img[i,j] = rgb_gamma2(accum_color / n_samples)
	end
	img
end

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
render(scene_2_spheres(), default_camera(), 96, 16)

render(scene_4_spheres(), default_camera(), 96, 16)

#md"""# Dielectrics

# from Section 10.2 Snell's Law:
# ![Ray refraction](https://raytracing.github.io/images/fig-1.13-refraction.jpg)

# Refracted angle `sinθ′ = (η/η′)⋅sinθ`, where η (\eta) are the refractive indices.

# Split the parts of the ray into `R′=R′⊥+R′∥` (perpendicular and parallel to n⃗′)."""

# """
# 	Args:
# 		refraction_ratio: incident refraction index divided by refraction index of 
# 			hit surface. i.e. η/η′ in the figure above"""
function refract(dir::SVector{3,Float32}, n⃗::SVector{3,Float32}, refraction_ratio::Float32)
	cosθ = min(-dir ⋅ n⃗, 1)
	r_out_perp = refraction_ratio * (dir + cosθ*n⃗)
	r_out_parallel = -√(abs(1-squared_length(r_out_perp))) * n⃗
	normalize(r_out_perp + r_out_parallel)
end

@assert refract(SVector{3,Float32}(0.6,-0.8,0), SVector{3,Float32}(0,1,0), 1.0f0) == SVector{3,Float32}(0.6,-0.8,0) # unchanged

t_refract_widerθ = refract(SVector{3,Float32}(0.6,-0.8,0), SVector{3,Float32}(0,1,0), 2.0f0) # wider angle
@assert isapprox(t_refract_widerθ, SVector{3,Float32}(0.87519, -0.483779, 0.0); atol=1e-3)

t_refract_narrowerθ = refract(SVector{3,Float32}(0.6,-0.8,0), SVector{3,Float32}(0,1,0), 0.5f0) # narrower angle
@assert isapprox(t_refract_narrowerθ, SVector{3,Float32}(0.3, -0.953939, 0.0); atol=1e-3)

mutable struct Dielectric <: Material
	ir::Float32 # index of refraction, i.e. η.
end

function reflectance(cosθ::Float32, refraction_ratio::Float32)
	# Use Schlick's approximation for reflectance.
	# claforte: may be buggy? I'm getting black pixels in the Hollow Glass Sphere...
	r0 = (1f0-refraction_ratio) / (1f0+refraction_ratio)
	r0 = r0^2
	r0 + (1f0-r0)*((1f0-cosθ)^5)
end

function scatter(mat::Dielectric, r_in::Ray, rec::HitRecord)
	attenuation = @SVector[1f0,1f0,1f0]
	refraction_ratio = rec.front_face ? (1.0f0/mat.ir) : mat.ir # i.e. ηᵢ/ηₜ
	cosθ = min(-r_in.dir⋅rec.n⃗, 1.0f0)
	sinθ = √(1.0f0 - cosθ^2)
	cannot_refract = refraction_ratio * sinθ > 1.0
	if cannot_refract || reflectance(cosθ, refraction_ratio) > rand()
		dir = reflect(r_in.dir, rec.n⃗)
	else
		dir = refract(r_in.dir, rec.n⃗, refraction_ratio)
	end
	Scatter(Ray(rec.p, dir), attenuation) # TODO: rename reflected -> !absorbed?
end

#"From C++: Image 15: Glass sphere that sometimes refracts"
function scene_diel_spheres(left_radius=0.5f0)::HittableList # dielectric spheres
	spheres = Sphere[]
	
	# small center sphere
	push!(spheres, Sphere(SVector{3,Float32}(0f0,0f0,-1f0), 0.5f0, Lambertian(SVector{3,Float32}(0.1f0,0.2f0,0.5f0))))
	
	# ground sphere (planet?)
	push!(spheres, Sphere(SVector{3,Float32}(0f0,-100.5f0,-1f0), 100f0, Lambertian(SVector{3,Float32}(0.8f0,0.8f0,0.0f0))))
	
	# left and right spheres.
	# Use a negative radius on the left sphere to create a "thin bubble" 
	push!(spheres, Sphere(SVector{3,Float32}(-1f0,0f0,-1f0), left_radius, Dielectric(1.5f0))) 
	push!(spheres, Sphere(SVector{3,Float32}( 1f0,0f0,-1f0), 0.5f0, Metal(SVector{3,Float32}(0.8f0,0.6f0,0.2f0), 0.0f0)))
	HittableList(spheres)
end

render(scene_diel_spheres(), default_camera(), 96, 16)
#render(scene_diel_spheres(), default_camera(), 320, 32)

#md"# Positioning camera"

function scene_blue_red_spheres()::HittableList # dielectric spheres
	spheres = Sphere[]
	R = cos(pi/4)
	push!(spheres, Sphere(@SVector[-R,0f0,-1f0], R, Lambertian(@SVector[0f0,0f0,1f0]))) 
	push!(spheres, Sphere(@SVector[R,0f0,-1f0], R, Lambertian(@SVector[1f0,0f0,0f0]))) 
	HittableList(spheres)
end

#md"# Spheres with depth-of-field"

#md"# Random spheres"

function scene_random_spheres()::HittableList # dielectric spheres
	spheres = Sphere[]

	# ground 
	push!(spheres, Sphere(@SVector[0f0,-1000f0,-1f0], 1000f0, 
						  Lambertian(@SVector[0.5f0,0.5f0,0.5f0])))

	for a in -11:10, b in -11:10
		choose_mat = rand()
		center = @SVector[a + 0.9f0*rand(), 0.2f0, b + 0.90f0*rand()]

		# skip spheres too close?
		if norm(center - @SVector[4f0,0.2f0,0f0]) < 0.9f0 continue end 
			
		if choose_mat < 0.8f0
			# diffuse
			albedo = @SVector[rand() for i ∈ 1:3] .* @SVector[rand() for i ∈ 1:3]
			push!(spheres, Sphere(center, 0.2f0, Lambertian(albedo)))
		elseif choose_mat < 0.95f0
			# metal
			albedo = @SVector[random_between(0.5f0,1.0f0) for i ∈ 1:3]
			fuzz = random_between(0.0f0, 5.0f0)
			push!(spheres, Sphere(center, 0.2f0, Metal(albedo, fuzz)))
		else
			# glass
			push!(spheres, Sphere(center, 0.2f0, Dielectric(1.5f0)))
		end
	end

	push!(spheres, Sphere(@SVector[0f0,1f0,0f0], 1.0f0, Dielectric(1.5f0)))
	push!(spheres, Sphere(@SVector[-4f0,1f0,0f0], 1.0f0, 
						  Lambertian(@SVector[0.4f0,0.2f0,0.1f0])))
	push!(spheres, Sphere(@SVector[4f0,1f0,0f0], 1.0f0, 
						  Metal(@SVector[0.7f0,0.6f0,0.5f0], 0.0f0)))
	HittableList(spheres)
end

scene_random_spheres()

# Hollow Glass sphere using a negative radius
# claforte: getting a weird black halo in the glass sphere... might be due to my
# "fix" for previous black spots, by moving the RecordHit point a bit away from 
# the hit surface... 
render(scene_diel_spheres(-0.5f0), default_camera(), 96, 16)

render(scene_blue_red_spheres(), default_camera(), 96, 16)

render(scene_diel_spheres(), default_camera(@SVector[-2f0,2f0,1f0], @SVector[0f0,0f0,-1f0],
							 				@SVector[0f0,1f0,0f0], 20.0f0), 96, 16)

t_lookfrom1 = @SVector[13.0f0,2.0f0,3.0f0]
t_lookat1 = @SVector[0.0f0,0.0f0,0.0f0]
t_cam1 = default_camera(t_lookfrom1, t_lookat1, @SVector[0.0f0,1.0f0,0.0f0], 20.0f0, 16.0f0/9.0f0,
                        0.1f0, 10.0f0)

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
render(scene_random_spheres(), t_cam1, 96, 1)

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
render(scene_random_spheres(), t_cam1, 200, 32) 

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
#   TBA
#render(scene_random_spheres(), t_cam, 1920, 1000)

t_lookfrom2 = @SVector[3.0f0,3.0f0,2.0f0]
t_lookat2 = @SVector[0.0f0,0.0f0,-1.0f0]
dist_to_focus2 = norm(t_lookfrom2-t_lookat2)
t_cam2 = default_camera(t_lookfrom2, t_lookat2, @SVector[0.0f0,1.0f0,0.0f0], 20.0f0, 16.0f0/9.0f0,
                        2.0f0, dist_to_focus2)

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
render(scene_diel_spheres(), t_cam2, 96, 16)

# using Profile
# Profile.clear_malloc_data()
# render(scene_2_spheres(), default_camera(), 96, 16)

