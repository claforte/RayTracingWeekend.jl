# Prototype - copied from pluto_RayTracingWeekend.jl
# Adapted from [Ray Tracing In One Weekend by Peter Shirley](https://raytracing.github.io/books/RayTracingInOneWeekend.html) and 
# [cshenton's Julia implementation](https://github.com/cshenton/RayTracing.jl)"
using Pkg
Pkg.activate(@__DIR__)

using BenchmarkTools, Images, InteractiveUtils, LinearAlgebra, StaticArrays
Option{T} = Union{Missing, T}
Vec3 = SVector{3}
Vec2 = SVector{2}
Point = Vec3
Color = Vec3
t_col = Color(0.4, 0.5, 0.1) # test color

# claforte: This was meant to be a convenient function to get some_vec.x or some_color.r,
# but this causes ~41 allocations per call, so this become a huge bottleneck.
#
# import Base.getproperty
# function Base.getproperty(vec::SVector{3}, sym::Symbol)
#     #  TODO: use a dictionary that maps symbols to indices, e.g. Dict(:x->1)
#     if sym in [:x, :r]
#         return vec[1]
#     elseif sym in [:y, :g]
#         return vec[2]
#     elseif sym in [:z, :b]
#         return vec[3]
#     else
#         return getfield(vec, sym)
#     end
# end

squared_length(v::SVector) = v ⋅ v

@btime squared_length(t_col)

squared_length(Color(0.4, 0.5, 0.1))

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
#
@btime squared_length(t_col)

near_zero(v::SVector) = squared_length(v) < 1e-5

t_col[1]# t_col.r
t_col[2] #t_col.y



# Test images

img = rand(4, 3)

img_rgb = rand(RGB{Float32}, 4, 4)

ex1 = [RGB{Float32}(1,0,0) RGB{Float32}(0,1,0) RGB{Float32}(0,0,1);
       RGB{Float32}(1,1,0) RGB{Float32}(1,1,1) RGB{Float32}(0,0,0)]

ex2 = zeros(RGB{Float32}, 2, 3)
ex2[1,1] = RGB{Float32}(1,0,0)
ex2

# TODO: save as image, e.g. PNG

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

rgb(v::Vec3) = RGB(v...)
rgb(t_col)

rgb_gamma2(v::Vec3) = RGB(sqrt.(v)...)

rgb_gamma2(t_col)

struct Ray
	origin::Point
	dir::Vec3 # direction (unit vector)
end

# equivalent to C++'s ray.at()
function point(r::Ray, t::AbstractFloat)::Point # point at parameter t
	r.origin .+ t .* r.dir
end

#md"# Chapter 4: Rays, simple camera, and background"

function sky_color(ray::Ray)
	# NOTE: unlike in the C++ implementation, we normalize the ray direction.
	t = 0.5(ray.dir[2] + 1.0)
    #t = 0.5(ray.dir.y + 1.0)
	(1-t)*Color(1,1,1) + t*Color(0.5, 0.7, 1.0)
end

# interpolates between blue and white
rgb(Color(0.5, 0.7, 1.0)), rgb(Color(1.0, 1.0, 1.0))

rgb(sky_color(Ray(Point(0,0,0), Vec3(0,-1,0))))

# md"""# Random vectors
# C++'s section 8.1"""

random_between(min=0.0, max=1.0) = rand()*(max-min) + min # equiv to random_double()
#random_between(50, 100)

#[random_between(50.0, 100.0) for i in 1:3]

function random_vec3(min=0.0, max=1.0)
	Vec3([random_between(min, max) for i in 1:3]...)
end

random_vec3(-1,1)

function random_vec3_in_sphere() # equiv to random_in_unit_sphere()
	while (true)
		p = random_vec3(-1,1)
		if squared_length(p) <= 1
			return p
		end
	end
end

squared_length(random_vec3_in_sphere())

"Random unit vector. Equivalent to C++'s `unit_vector(random_in_unit_sphere())`"
random_vec3_on_sphere() = normalize(random_vec3_in_sphere())
random_vec3_on_sphere()
norm(random_vec3_on_sphere())

function random_vec2_in_disk() :: Vec2 # equiv to random_in_unit_disk()
	while (true)
		p = Vec2(rand(2)...)
		if squared_length(p) <= 1
			return p
		end
	end
end


""" Temporary function to shoot rays through each pixel. Later replaced by `render`
	
	Args:
		scene: a function that takes a ray, returns the color of any object it hit
"""
function main(nx::Int, ny::Int, scene)
	lower_left_corner = Point(-2, -1, -1)
	horizontal = Vec3(4, 0, 0)
	vertical = Vec3(0, 2, 0)
	origin = Point(0, 0, 0)
	
	img = zeros(RGB, ny, nx)
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

main(200,100, sky_color)

#md"# Chapter 5: Add a sphere"

function hit_sphere1(center::Vec3, radius::AbstractFloat, r::Ray)
	oc = r.origin - center
	a = r.dir ⋅ r.dir
	b = 2oc ⋅ r.dir
	c = (oc ⋅ oc) - radius*radius
	discriminant = b*b - 4a*c
	discriminant > 0
end

function sphere_scene1(r::Ray)
	if hit_sphere1(Vec3(0,0,-1), 0.5, r) # sphere of radius 0.5 centered at z=-1
		return Vec3(1,0,0) # red
	else
		sky_color(r)
	end
end

main(200,100,sphere_scene1)

#md"# Chapter 6: Surface normals and multiple objects"

function hit_sphere2(center::Vec3, radius::AbstractFloat, r::Ray)
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
	sphere_center = Vec3(0,0,-1)
	t = hit_sphere2(sphere_center, 0.5, r) # sphere of radius 0.5 centered at z=-1
	if t > 0
		n⃗ = normalize(point(r, t) - sphere_center) # normal vector. typed n\vec
		return 0.5n⃗ + Vec3(0.5,0.5,0.5) # remap normal to rgb
	else
		sky_color(r)
	end
end

main(200,100,sphere_scene2)

"An object that can be hit by Ray"
abstract type Hittable end

"""Materials tell us how rays interact with a surface"""
abstract type Material end

"Record a hit between a ray and an object's surface"
mutable struct HitRecord
	# claforte: Not sure if this needs to be mutable... might impact performance!

	t::Float64 # vector from the ray's origin to the intersection with a surface
	p::Vec3 # point of the intersection between an object's surface and a ray
	n⃗::Vec3 # surface's outward normal vector, points towards outside of object?
	
	# If true, our ray hit from outside to the front of the surface. 
	# If false, the ray hit from within.
	front_face::Bool
	mat::Material
end

struct Sphere <: Hittable
	center::Vec3
	radius::Float64
	mat::Material
end

# md"""The geometry defines an `outside normal`. A HitRecord stores the `local normal`.
# ![Surface normal](https://raytracing.github.io/images/fig-1.06-normal-sides.jpg)
# """

"""Equivalent to `hit_record.set_face_normal()`"""
function ray_to_HitRecord(t, p, outward_n⃗, r_dir::Vec3, mat::Material)
	front_face = r_dir ⋅ outward_n⃗ < 0
	n⃗ = front_face ? outward_n⃗ : -outward_n⃗
	rec = HitRecord(t,p,n⃗,front_face,mat)
end

struct Scatter
	r::Ray
	attenuation::Color
	
	# claforte: TODO: rename to "absorbed?", i.e. not reflected/refracted?
	reflected::Bool # whether the scattered ray was reflected, or fully absorbed
	Scatter(r,a,reflected=true) = new(r,a,reflected)
end

#"Diffuse material"
mutable struct Lambertian<:Material
	albedo::Color
end

"""Compute reflection vector for v (pointing to surface) and normal n⃗.

	See [diagram](https://raytracing.github.io/books/RayTracingInOneWeekend.html#metal/mirroredlightreflection)"""
reflect(v::Vec3, n⃗::Vec3) = normalize(v - (2v⋅n⃗)*n⃗) # claforte: normalize not needed?

@assert reflect(Vec3(0.6,-0.8,0), Vec3(0,1,0)) == Vec3(0.6,0.8,0) # diagram's example

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

function hit(s::Sphere, r::Ray, tmin::Float64, tmax::Float64)::Option{HitRecord}
    oc = r.origin - s.center
    a = 1 #r.dir ⋅ r.dir # normalized vector - always 1
    half_b = oc ⋅ r.dir
    c = oc⋅oc - s.radius^2
    discriminant = half_b^2 - a*c
	if discriminant < 0 return missing end
	sqrtd = √discriminant
	
	# Find the nearest root that lies in the acceptable range
	root = (-half_b - sqrtd) / a	
	if root < tmin || tmax < root
		root = (-half_b + sqrtd) / a
		if root < tmin || tmax < root
			return missing
		end
	end
		
	t = root
	p = point(r, t)
	n⃗ = (p - s.center) / s.radius
	return ray_to_HitRecord(t, p, n⃗, r.dir,s.mat)
end

struct HittableList <: Hittable
    list::Vector{Hittable}
end

#"""Find closest hit between `Ray r` and a list of Hittable objects `h`, within distance `tmin` < `tmax`"""
function hit(hittables::HittableList, r::Ray, tmin::Float64,
			 tmax::Float64)::Option{HitRecord}
    closest = tmax # closest t so far
    rec = missing
    for h in hittables.list
        temprec = hit(h, r, tmin, closest)
        if !ismissing(temprec)
            rec = temprec
            closest = rec.t # i.e. ignore any further hit > this one's.
        end
    end
    rec
end

color_vec3_in_rgb(v::Vec3) = 0.5normalize(v) + Vec3(0.5,0.5,0.5)

#md"# Metal material"

mutable struct Metal<:Material
	albedo::Color
	fuzz::Float64 # how big the sphere used to generate fuzzy reflection rays. 0=none
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
	push!(spheres, Sphere(Vec3(0,0,-1), 0.5, Lambertian(Color(0.7,0.3,0.3))))
	
	# ground sphere (planet?)
	push!(spheres, Sphere(Vec3(0,-100.5,-1), 100, Lambertian(Color(0.8,0.8,0.0))))
	HittableList(spheres)
end

#"""Scene with 2 Lambertian, 2 Metal spheres.
#
#	See https://raytracing.github.io/images/img-1.11-metal-shiny.png"""
function scene_4_spheres()::HittableList
	scene = scene_2_spheres()

	# left and right Metal spheres
	push!(scene.list, Sphere(Vec3(-1,0,-1), 0.5, Metal(Color(0.8,0.8,0.8), 0.3))) 
	push!(scene.list, Sphere(Vec3( 1,0,-1), 0.5, Metal(Color(0.8,0.6,0.2), 0.8)))
	return scene
end

#md"""# Camera

# Adapted from C++'s sections 7.2, 11.1 """
mutable struct Camera
	origin::Vec3
	lower_left_corner::Vec3
	horizontal::Vec3
	vertical::Vec3
	u::Vec3
	v::Vec3
	w::Vec3
	lens_radius::Float64
end

"""
	Args:
		vfov: vertical field-of-view in degrees
		aspect_ratio: horizontal/vertical ratio of pixels
      aperture: if 0 - no depth-of-field
"""
function default_camera(lookfrom::Point=Point(0,0,0), lookat::Point=Point(0,0,-1), 
						vup::Vec3=Vec3(0,1,0), vfov=90.0, aspect_ratio=16.0/9.0,
						aperture=0.0, focus_dist=1.0)
	viewport_height = 2.0 * tand(vfov/2)
	viewport_width = aspect_ratio * viewport_height
	
	w = normalize(lookfrom - lookat)
	u = normalize(vup × w)
	v = w × u
	
	origin = lookfrom
	horizontal = focus_dist * viewport_width * u
	vertical = focus_dist * viewport_height * v
	lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w
	lens_radius = aperture/2
	Camera(origin, lower_left_corner, horizontal, vertical, u, v, w, lens_radius)
end

default_camera()

clamp(3.5, 0, 1)

#md"# Render

function get_ray(c::Camera, s::Float64, t::Float64)
	rd = Vec2(c.lens_radius * random_vec2_in_disk())
	offset = c.u * rd[1] + c.v * rd[2] #offset = c.u * rd.x + c.v * rd.y
    Ray(c.origin + offset, normalize(c.lower_left_corner + s*c.horizontal +
									 t*c.vertical - c.origin - offset))
end

get_ray(default_camera(), 0.0, 0.0)

"""Compute color for a ray, recursively

	Args:
		depth: how many more levels of recursive ray bounces can we still compute?"""
function ray_color(r::Ray, world::HittableList, depth=4)::Vec3
    if depth <= 0
		return Vec3(0,0,0)
	end
		
	rec = hit(world, r, 1e-4, Inf)
    if !ismissing(rec)
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
			return Vec3(0,0,0)
		end
        # if s.reflected && depth < 20
        #     return s.attenuation .* color(s.ray, world, depth+1)
        # else
        #     return Vec3(0.0, 0.0, 0.0)
        # end
    else
        sky_color(r)
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
	aspect_ratio = 16.0/9.0 # TODO: use cam.aspect_ratio for consistency
	image_height = convert(Int64, floor(image_width / aspect_ratio))

	# Render
	img = zeros(RGB, image_height, image_width)
	# Compared to C++, Julia is:
	# 1. column-major, i.e. iterate 1 column at a time, so invert i,j compared to C++
	# 2. 1-based, so no need to subtract 1 from image_width, etc.
	# 3. The array is Y-down, but `v` is Y-up 
	for i in 1:image_height, j in 1:image_width
		accum_color = Vec3(0,0,0)
		for s in 1:n_samples
			u = j/image_width
			v = (image_height-i)/image_height # i is Y-down, v is Y-up!
			if s != 1 # 1st sample is always centered, for 1-sample/pixel
				# claforte: I think the C++ version had a bug, the rand offset was
				# between [0,1] instead of centered at 0, e.g. [-0.5, 0.5].
				u += (rand()-0.5) / image_width
				v += (rand()-0.5) / image_height
			end
			ray = get_ray(cam, u, v)
			accum_color += ray_color(ray, scene)
		end
		img[i,j] = rgb_gamma2(accum_color / n_samples)
	end
	img
end

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
function refract(dir::Vec3, n⃗::Vec3, refraction_ratio::Float64)
	cosθ = min(-dir ⋅ n⃗, 1)
	r_out_perp = refraction_ratio * (dir + cosθ*n⃗)
	r_out_parallel = -√(abs(1-squared_length(r_out_perp))) * n⃗
	normalize(r_out_perp + r_out_parallel)
end

@assert refract(Vec3(0.6,-0.8,0), Vec3(0,1,0), 1.0) == Vec3(0.6,-0.8,0) # unchanged

t_refract_widerθ = refract(Vec3(0.6,-0.8,0), Vec3(0,1,0), 2.0) # wider angle
@assert isapprox(t_refract_widerθ, Vec3(0.87519, -0.483779, 0.0); atol=1e-3)

t_refract_narrowerθ = refract(Vec3(0.6,-0.8,0), Vec3(0,1,0), 0.5) # narrower angle
@assert isapprox(t_refract_narrowerθ, Vec3(0.3, -0.953939, 0.0); atol=1e-3)

mutable struct Dielectric <: Material
	ir::Float64 # index of refraction, i.e. η.
end

function reflectance(cosθ::Float64, refraction_ratio::Float64)
	# Use Schlick's approximation for reflectance.
	# claforte: may be buggy? I'm getting black pixels in the Hollow Glass Sphere...
	r0 = (1-refraction_ratio) / (1+refraction_ratio)
	r0 = r0^2
	r0 + (1-r0)*((1-cosθ)^5)
end

function scatter(mat::Dielectric, r_in::Ray, rec::HitRecord)
	attenuation = Color(1,1,1)
	refraction_ratio = rec.front_face ? (1.0/mat.ir) : mat.ir # i.e. ηᵢ/ηₜ
	cosθ = min(-r_in.dir⋅rec.n⃗, 1.0)
	sinθ = √(1.0 - cosθ^2)
	cannot_refract = refraction_ratio * sinθ > 1.0
	if cannot_refract || reflectance(cosθ, refraction_ratio) > rand()
		dir = reflect(r_in.dir, rec.n⃗)
	else
		dir = refract(r_in.dir, rec.n⃗, refraction_ratio)
	end
	Scatter(Ray(rec.p, dir), attenuation) # TODO: rename reflected -> !absorbed?
end

#"From C++: Image 15: Glass sphere that sometimes refracts"
function scene_diel_spheres(left_radius=0.5)::HittableList # dielectric spheres
	spheres = Sphere[]
	
	# small center sphere
	push!(spheres, Sphere(Vec3(0,0,-1), 0.5, Lambertian(Color(0.1,0.2,0.5))))
	
	# ground sphere (planet?)
	push!(spheres, Sphere(Vec3(0,-100.5,-1), 100, Lambertian(Color(0.8,0.8,0.0))))
	
	# left and right spheres.
	# Use a negative radius on the left sphere to create a "thin bubble" 
	push!(spheres, Sphere(Vec3(-1,0,-1), left_radius, Dielectric(1.5))) 
	push!(spheres, Sphere(Vec3( 1,0,-1), 0.5, Metal(Color(0.8,0.6,0.2), 0.0)))
	HittableList(spheres)
end

render(scene_diel_spheres(), default_camera(), 96, 16)
#render(scene_diel_spheres(), default_camera(), 320, 32)

#md"# Positioning camera"

function scene_blue_red_spheres()::HittableList # dielectric spheres
	spheres = Sphere[]
	R = cos(pi/4)
	push!(spheres, Sphere(Vec3(-R,0,-1), R, Lambertian(Color(0,0,1)))) 
	push!(spheres, Sphere(Vec3( R,0,-1), R, Lambertian(Color(1,0,0)))) 
	HittableList(spheres)
end

#md"# Spheres with depth-of-field"

#md"# Random spheres"



function scene_random_spheres()::HittableList # dielectric spheres
	spheres = Sphere[]

	# ground 
	push!(spheres, Sphere(Vec3(0,-1000,-1), 1000, Lambertian(Color(0.5,0.5,0.5))))

	for a in -11:10, b in -11:10
		choose_mat = rand()
		center = Point(a + 0.9*rand(), 0.2, b + 0.9*rand())
		
		if norm(center - Point(4,0.2,0)) < 0.9 continue end # skip spheres too close?
			
		if choose_mat < 0.8
			# diffuse
			albedo = Color(rand(3)...) .* Color(rand(3)...) # TODO: random_color()
			push!(spheres, Sphere(center, 0.2, Lambertian(albedo)))
		elseif choose_mat < 0.95
			# metal
			albedo = Color(random_between(0.5,1.0), random_between(0.5,1.0),
						   random_between(0.5,1.0)) # TODO: random_color
			fuzz = random_between(0.0, 5.0)
			push!(spheres, Sphere(center, 0.2, Metal(albedo, fuzz)))
		else
			# glass
			push!(spheres, Sphere(center, 0.2, Dielectric(1.5)))
		end
	end

	push!(spheres, Sphere(Point(0,1,0), 1.0, Dielectric(1.5)))
	push!(spheres, Sphere(Point(-4,1,0), 1.0, Lambertian(Color(0.4,0.2,0.1))))
	push!(spheres, Sphere(Point(4,1,0), 1.0, Metal(Color(0.7,0.6,0.5), 0.0)))
	HittableList(spheres)
end

scene_random_spheres()

# Hollow Glass sphere using a negative radius
# claforte: getting a weird black halo in the glass sphere... might be due to my
# "fix" for previous black spots, by moving the RecordHit point a bit away from 
# the hit surface... 
render(scene_diel_spheres(-0.5), default_camera(), 96, 16)

render(scene_blue_red_spheres(), default_camera(), 96, 16)

render(scene_diel_spheres(), default_camera(Point(-2,2,1), Point(0,0,-1),
							 				Vec3(0,1,0), 20.0), 96, 16)

t_lookfrom2 = Point(13.0,2.0,3.0)
t_lookat2 = Point(0.0,0.0,0.0)
t_cam = default_camera(t_lookfrom2, t_lookat2, Vec3(0.0,1.0,0.0), 20.0, 16.0/9.0,
                        0.1, 10.0)

# took ~20s (really rough timing) in REPL, before optimization
# after optimization: 880.997 ms (48801164 allocations: 847.10 MiB)
render(scene_random_spheres(), t_cam, 96, 1)
#render(scene_random_spheres(), t_cam, 200, 32) # took 5020s in Pluto.jl, before optimizations!

t_lookfrom = Point(3.0,3.0,2.0)
t_lookat = Point(0.0,0.0,-1.0)
dist_to_focus = norm(t_lookfrom-t_lookat)
t_cam = default_camera(t_lookfrom, t_lookat, Vec3(0.0,1.0,0.0), 20.0, 16.0/9.0,
                        2.0, dist_to_focus)

# Before optimization:
#  5.993 s (193097930 allocations: 11.92 GiB)
# after disabling: `Base.getproperty(vec::SVector{3}, sym::Symbol)`
#  1.001 s ( 17406437 allocations: 425.87 MiB)
render(scene_diel_spheres(), t_cam, 96, 16)

