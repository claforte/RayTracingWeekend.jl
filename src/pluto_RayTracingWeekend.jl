### A Pluto.jl notebook ###
# v0.17.2

using Markdown
using InteractiveUtils

# ╔═╡ 38fdd4ef-c383-4f97-8451-c6f602307e7d
using Images

# ╔═╡ 3dceca5d-7d1e-425b-9516-24e0a24adaff
using LinearAlgebra

# ╔═╡ 3eb50f44-9091-45e8-a7e1-92d25b4b2090
begin
	using StaticArrays
	Option{T} = Union{Missing, T}
	Vec3 = SVector{3}
	Point = Vec3
	Color = Vec3
	
	import Base.getproperty
	function Base.getproperty(vec::Vec3, sym::Symbol)
		#  TODO: use a dictionary that maps symbols to indices, e.g. Dict(:x->1)
		if sym in [:x, :r]
			return vec[1]
		elseif sym in [:y, :g]
			return vec[2]
		elseif sym in [:z, :b]
			return vec[3]
		else
			return getfield(vec, sym)
		end
	end

	squared_length(vec::Vec3) = vec ⋅ vec
	near_zero(vec::Vec3) = squared_length(vec) < 1e-5
end

# ╔═╡ 0866add2-9b95-45e7-8081-c01cd2a66911
md"""# Chapter 1: Overview"""

# ╔═╡ 611d5eae-4b09-11ec-27bf-ef4a1ecdcc41
md"Adapted from [Ray Tracing In One Weekend by Peter Shirley](https://raytracing.github.io/books/RayTracingInOneWeekend.html) and [cshenton's Julia implementation](https://github.com/cshenton/RayTracing.jl)"

# ╔═╡ 97bb4432-ed41-423b-b4d9-bafc519de641
md"# Chapter 2 - Manipulating images"

# ╔═╡ 84d29423-cf11-41c3-af4d-c5f63b1ef23e
img = rand(4, 3)

# ╔═╡ 7d26fef0-9a06-479a-ae08-e9d04e455767
img_rgb = rand(RGB{Float32}, 4, 4)

# ╔═╡ f29ad2c0-c3ff-484d-8fdd-dff34d2bb863
ex1 = [RGB{Float32}(1,0,0) RGB{Float32}(0,1,0) RGB{Float32}(0,0,1);
       RGB{Float32}(1,1,0) RGB{Float32}(1,1,1) RGB{Float32}(0,0,0)]

# ╔═╡ 2192e695-4378-4b47-8ce0-353636cd2cd1
begin
	ex2 = zeros(RGB{Float32}, 2, 3)
	ex2[1,1] = RGB{Float32}(1,0,0)
	ex2
end

# ╔═╡ 538d1aa5-07f9-4fca-8410-ef63b8a6857b
# TODO: save as image, e.g. PNG

# ╔═╡ 8aeb7373-6bb0-4544-8655-fa941561688c
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

# ╔═╡ 154d736b-8fdc-44af-ae2a-9e5ba6d2c92e
gradient(200,100)

# ╔═╡ 216922d8-613a-4ac1-9559-40878e6587e2
md"""Unlike the C++ implementation:
- Julia uses i for row, j for column, so I inverted the C++ code's variable names.
- The C++ code used a Y-up coordinate system, so I used (ny-i) instead of i for the row number."""

# ╔═╡ 961fd749-d439-4dfa-ae21-b1659dc54511
md"# Chapter 3: Linear Algebra"

# ╔═╡ 7249557b-043c-4994-bec7-615f137f98e3
md"""See https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/

Use these convenient unicode characters:
- \times: ×
- \cdot: ⋅
"""

# ╔═╡ f8007c75-9487-414a-9592-138a696c2957
# Dot product (\cdot) is defined by LinearAlgebra...
[1; 1] ⋅ [2; 3]

# ╔═╡ 668030c8-24a7-4aa6-b858-cedf8ac5f988
# Cross product (\times)
[0;1;0] × [0;0;1]

# ╔═╡ 78f209df-d176-4711-80fc-a8054771f105
t_col = Color(0.4, 0.5, 0.1) # test color

# ╔═╡ e88de775-6904-4182-8209-06db22758470
t_col.r

# ╔═╡ 3e6fd5c0-6d4a-44ef-a7b2-106b52fc6550
t_col.y

# ╔═╡ 5fd1ec87-3616-448a-ab4d-fede804b26d5
squared_length(t_col)

# ╔═╡ a0893bf4-9607-4853-8162-9f34d3337060
rgb(v::Vec3) = RGB(v.r, v.g, v.b)

# ╔═╡ cfbcb883-d12e-4ad3-a084-064749bddcdb
rgb(t_col)

# ╔═╡ 6348c03d-e8ec-4dbb-9f8a-8e4a48bb1cb3
rgb_gamma2(v::Vec3) = RGB(sqrt.(v)...)

# ╔═╡ dbc8fc2d-39c2-4ec9-b82d-7c6a8b12dde7
rgb_gamma2(t_col)

# ╔═╡ 53832af1-a9be-4e02-8b71-a70dae63c233
struct Ray
	origin::Point
	dir::Vec3 # direction (unit vector)
end

# ╔═╡ 81b4c9e4-9f93-45ca-9fa0-7e9686a55e9a
# equivalent to C++'s ray.at()
function point(r::Ray, t::AbstractFloat)::Point # point at parameter t
	r.origin .+ t .* r.dir
end

# ╔═╡ 678214c5-de81-489f-b002-c343d48071c9
md"# Chapter 4: Rays, simple camera, and background"

# ╔═╡ cbb6418c-79e9-4359-80a6-40a8fa40679e
function sky_color(ray::Ray)
	# NOTE: unlike in the C++ implementation, we normalize the ray direction.
	t = 0.5(ray.dir.y + 1.0)
	(1-t)*Color(1,1,1) + t*Color(0.5, 0.7, 1.0)
end

# ╔═╡ 14be6068-6a15-4fad-ac0a-f156da71f103
# interpolates between blue and white
rgb(Color(0.5, 0.7, 1.0)), rgb(Color(1.0, 1.0, 1.0))

# ╔═╡ 64ef0313-2d2b-49d5-a1a1-3b04426a82f8
begin
	rgb(sky_color(Ray(Point(0,0,0), Vec3(0,-1,0))))
end

# ╔═╡ 971777a6-f269-4344-8dba-7a55118396e5
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

# ╔═╡ 655ffa6c-f1e9-4149-8f1d-51145c5a51e4
main(200,100, sky_color)

# ╔═╡ 9075f8ed-f319-486d-94b2-486806aba3fd
md"# Chapter 5: Add a sphere"

# ╔═╡ c00e2004-2002-4dd2-98ed-f898ef2c14f1
function hit_sphere1(center::Vec3, radius::AbstractFloat, r::Ray)
	oc = r.origin - center
	a = r.dir ⋅ r.dir
	b = 2oc ⋅ r.dir
	c = (oc ⋅ oc) - radius*radius
	discriminant = b*b - 4a*c
	discriminant > 0
end

# ╔═╡ b7399fb8-6205-41ea-9c70-eb62daedcefb
function sphere_scene1(r::Ray)
	if hit_sphere1(Vec3(0,0,-1), 0.5, r) # sphere of radius 0.5 centered at z=-1
		return Vec3(1,0,0) # red
	else
		sky_color(r)
	end
end

# ╔═╡ 1d04159d-87bd-4cf8-a73c-817f20ca1026
main(200,100,sphere_scene1)

# ╔═╡ fed81f09-e4a0-433f-99e8-261096114b7b
md"# Chapter 6: Surface normals and multiple objects"

# ╔═╡ 24e8740a-8e44-4206-b2b6-c4a55002dad8
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

# ╔═╡ 359832af-7598-4c45-8033-c28cb0d86772
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

# ╔═╡ ed6ab8be-587c-4cb6-8172-618c74d3f9cc
main(200,100,sphere_scene2)

# ╔═╡ a65c68c9-e489-465a-9687-93ae9da14a5e
"An object that can be hit by Ray"
abstract type Hittable end

# ╔═╡ 2c4b4453-1a46-4889-9a14-16b18cc8c240
"""Materials tell us how rays interact with a surface"""
abstract type Material end

# ╔═╡ 98c43f3f-4bfc-49db-806a-850b7d75b5a4


# ╔═╡ 3b570d37-f407-41d8-b8a0-a0af4d85b14d
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

# ╔═╡ 138bb5b6-0f45-4f13-8339-5110eb7cd1ff
struct Sphere <: Hittable
	center::Vec3
	radius::Float64
	mat::Material
end

# ╔═╡ 6b36d245-bf01-45a7-b119-8315226dd4a3
md"""The geometry defines an `outside normal`. A HitRecord stores the `local normal`.
![Surface normal](https://raytracing.github.io/images/fig-1.06-normal-sides.jpg)
"""

# ╔═╡ 4a396b3f-f920-4ec2-91f6-7d61fe2b9699
"""Equivalent to `hit_record.set_face_normal()`"""
function ray_to_HitRecord(t, p, outward_n⃗, r_dir::Vec3, mat::Material)
	front_face = r_dir ⋅ outward_n⃗ < 0
	n⃗ = front_face ? outward_n⃗ : -outward_n⃗
	rec = HitRecord(t,p,n⃗,front_face,mat)
end

# ╔═╡ 427f247c-055c-459e-9862-26e9f6f3e24f
struct Scatter
	r::Ray
	attenuation::Color
	
	# claforte: TODO: rename to "absorbed?", i.e. not reflected/refracted?
	reflected::Bool # whether the scattered ray was reflected, or fully absorbed
	Scatter(r,a,reflected=true) = new(r,a,reflected)
end

# ╔═╡ 88e51c27-0f28-4dcc-b9e9-ac44eeb876f5
#"Diffuse material"
mutable struct Lambertian<:Material
	albedo::Color
end

# ╔═╡ 7c4a67b2-8208-4cd4-b1ea-16f6f50adfe8
"""Compute reflection vector for v (pointing to surface) and normal n⃗.

	See [diagram](https://raytracing.github.io/books/RayTracingInOneWeekend.html#metal/mirroredlightreflection)"""
reflect(v::Vec3, n⃗::Vec3) = normalize(v - (2v⋅n⃗)*n⃗) # claforte: normalize not needed?

# ╔═╡ ca649864-5a6d-4ca7-896e-e80e8a48443e
@assert reflect(Vec3(0.6,-0.8,0), Vec3(0,1,0)) == Vec3(0.6,0.8,0) # diagram's example

# ╔═╡ 78efebc5-53fd-417d-bd9e-667fd504e3fd
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

# ╔═╡ 05e57afd-6eb9-42c5-9666-7be3771fa6b8
struct HittableList <: Hittable
    list::Vector{Hittable}
end

# ╔═╡ 08e18ae5-9927-485e-9644-552f03e06f27
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

# ╔═╡ 737e2f87-82f5-45b6-a76c-4f560c29f5b9
color_vec3_in_rgb(v::Vec3) = 0.5normalize(v) + Vec3(0.5,0.5,0.5)

# ╔═╡ 0bf88264-d4c5-4d5a-babe-d2433e46024d
md"# Metal material"

# ╔═╡ 555bea1d-5178-48dd-87e6-4e2a2471a5dd
mutable struct Metal<:Material
	albedo::Color
	fuzz::Float64 # how big the sphere used to generate fuzzy reflection rays. 0=none
	Metal(a,f=0.0) = new(a,f)
end

# ╔═╡ 851c002c-dc23-4999-b28c-a716c5d2d42c
md"# Scenes"

# ╔═╡ 70530f8e-1b29-4588-927f-d38d5d12d5c9
#"Scene with 2 Lambertian spheres"
function scene_2_spheres()::HittableList
	spheres = Sphere[]
	
	# small center sphere
	push!(spheres, Sphere(Vec3(0,0,-1), 0.5, Lambertian(Color(0.7,0.3,0.3))))
	
	# ground sphere (planet?)
	push!(spheres, Sphere(Vec3(0,-100.5,-1), 100, Lambertian(Color(0.8,0.8,0.0))))
	HittableList(spheres)
end

# ╔═╡ c5349670-4df4-421f-9d5a-b28c1b9040c2
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

# ╔═╡ 282a4912-7a6e-44ae-90eb-f2f7c8f3d0f4
md"""# Camera

Adapted from C++'s section 7.2
"""

# ╔═╡ a0e5a1f3-244f-427b-a335-7e233af1d9d8
mutable struct Camera
	origin::Vec3
	lower_left_corner::Vec3
	horizontal::Vec3
	vertical::Vec3
end

# ╔═╡ 5d00f26b-35f2-4071-8e04-227ffc25f184
function default_camera(aspect_ratio=16.0/9.0, viewport_height=2.0, focal_length=1.0)
	viewport_width = aspect_ratio * viewport_height
	origin = Vec3(0,0,0)
	horizontal = Vec3(viewport_width, 0, 0)
	vertical = Vec3(0, viewport_height, 0)
	lower_left_corner = origin - horizontal/2 - vertical/2 - Vec3(0,0,focal_length)
	Camera(origin, lower_left_corner, horizontal, vertical)
end

# ╔═╡ c1aef1be-79d4-4417-be36-ae8416465986
default_camera()

# ╔═╡ 94081092-afc6-4359-bd2c-15e8407bf70e
get_ray(c::Camera, u::Float64, v::Float64) =
	Ray(c.origin, normalize(c.lower_left_corner + u*c.horizontal + v*c.vertical - c.origin))

# ╔═╡ 813eaa13-2eb2-4302-9e4d-5d1dab0ac7c4
get_ray(default_camera(), 0.0, 0.0)

# ╔═╡ a4493f3a-cb9f-404e-830d-a7d007df5baf
clamp(3.5, 0, 1)

# ╔═╡ 891ce2c8-f8b2-472b-a8d9-389dafddcf22
md"# Render

(equivalent to final `main`)"

# ╔═╡ 97d9a286-2e70-4dd4-8407-62b3a89da16b
md"2 spheres (1 sample per pixel, i.e. aliased):"

# ╔═╡ 4dd59aa7-37a7-426b-8573-a0fee26343df
#render(scene_2_spheres(), default_camera(), 96, 16)

# ╔═╡ 5f1bae02-d425-4a73-8668-d6383faba79d
md"""# Dielectrics

from Section 10.2 Snell's Law:
![Ray refraction](https://raytracing.github.io/images/fig-1.13-refraction.jpg)

Refracted angle `sinθ′ = (η/η′)⋅sinθ`, where η (\eta) are the refractive indices.

Split the parts of the ray into `R′=R′⊥+R′∥` (perpendicular and parallel to n⃗′)."""

# ╔═╡ 776b5096-4792-46b9-b70f-50a2b4c4999b


# ╔═╡ 9dc64353-c41c-45b4-aacd-12d5d6117c58
function refract(dir::Vec3, n⃗::Vec3, ηᵢ_over_ηₜ::Float64)
	cosθ = min(-dir ⋅ n⃗, 1)
	r_out_perp = ηᵢ_over_ηₜ * (dir + cosθ*n⃗) # claforte: can't use ⟂ (\perp)... sigh
	r_out_parallel = -√(abs(1-squared_length(r_out_perp))) * n⃗ # can't use ∥...
	normalize(r_out_perp + r_out_parallel)
end

# ╔═╡ e1adf736-592e-4980-9318-a641a2188b8a
@assert refract(Vec3(0.6,-0.8,0), Vec3(0,1,0), 1.0) == Vec3(0.6,-0.8,0) # unchanged

# ╔═╡ bbfd4db5-3650-4f27-9777-2ff981c3d28b
begin
	t_refract_widerθ = refract(Vec3(0.6,-0.8,0), Vec3(0,1,0), 2.0) # wider angle
	@assert isapprox(t_refract_widerθ, Vec3(0.87519, -0.483779, 0.0); atol=1e-3)
end

# ╔═╡ b54e631c-12ab-49c1-a63e-65b1c9a5d8b6
begin
	t_refract_narrowerθ = refract(Vec3(0.6,-0.8,0), Vec3(0,1,0), 0.5) # narrower angle
	@assert isapprox(t_refract_narrowerθ, Vec3(0.3, -0.953939, 0.0); atol=1e-3)
end

# ╔═╡ f5c4e502-048c-4fcd-879f-eaeb4430c012
mutable struct Dielectric <: Material
	ir::Float64 # index of refraction, i.e. η
end

# ╔═╡ ae3b8f15-985d-4f74-ac8c-86a3ffc3b8b1
function scatter(mat::Dielectric, r_in::Ray, rec::HitRecord)
	attenuation = Color(1,1,1)
	ηᵢ_over_ηₜ = rec.front_face ? (1.0/mat.ir) : mat.ir # refraction ratio
	cosθ = min(-r_in.dir⋅rec.n⃗, 1.0)
	sinθ = √(1.0 - cosθ^2)
	cannot_refract = ηᵢ_over_ηₜ * sinθ > 1.0
	if cannot_refract
		dir = reflect(r_in.dir, rec.n⃗)
	else
		dir = refract(r_in.dir, rec.n⃗, ηᵢ_over_ηₜ)
	end
	Scatter(Ray(rec.p, dir), attenuation) # TODO: rename reflected -> !absorbed?
end

# ╔═╡ ddf5883c-036a-4a21-908d-bb7cec731f7f
#"From C++: Image 15: Glass sphere that sometimes refracts"
function scene_diel_spheres()::HittableList # dielectric spheres
	spheres = Sphere[]
	
	# small center sphere
	push!(spheres, Sphere(Vec3(0,0,-1), 0.5, Lambertian(Color(0.1,0.2,0.5))))
	
	# ground sphere (planet?)
	push!(spheres, Sphere(Vec3(0,-100.5,-1), 100, Lambertian(Color(0.8,0.8,0.0))))
	
	# left and right Metal spheres
	push!(spheres, Sphere(Vec3(-1,0,-1), 0.5, Dielectric(1.5))) 
	push!(spheres, Sphere(Vec3( 1,0,-1), 0.5, Metal(Color(0.8,0.6,0.2), 0.0)))
	HittableList(spheres)
end

# ╔═╡ 30102751-fbbd-41dc-9dc1-5c7cb8cd613f
md"""# Random vectors

C++'s section 8.1"""


# ╔═╡ 1f4a9699-5c91-4e2c-b592-1bfa86c05959
random_between(min=0.0, max=1.0) = rand()*(max-min) + min # equiv to random_double()

# ╔═╡ 795fdf6f-945e-44f8-8aa1-1e33586cc095
random_between(50, 100)

# ╔═╡ f7d47388-076b-416c-a088-c8963009faa6
begin
	[random_between(50.0, 100.0) for i in 1:3]
end

# ╔═╡ 927e351c-3529-4dc1-b342-e5c040fdc6e2
function random_vec3(min=0.0, max=1.0)
	Vec3([random_between(min, max) for i in 1:3]...)
end

# ╔═╡ 7e98cea8-90b4-44da-b234-a60c1a3b2fa4
random_vec3(-1,1)

# ╔═╡ 58e91524-be98-4b67-bd9a-5c682cb5c009
function random_vec3_in_sphere() # equiv to random_in_unit_sphere()
	while (true)
		p = random_vec3(-1,1)
		if squared_length(p) <= 1
			return p
		end
	end
end

# ╔═╡ 44fc6b51-c682-4fa9-9ebe-804a4f3397be
squared_length(random_vec3_in_sphere())

# ╔═╡ a0c4f8dc-5580-413b-99ee-dcb7b6c92c8d
"Random unit vector. Equivalent to C++'s `unit_vector(random_in_unit_sphere())`"
random_vec3_on_sphere() = normalize(random_vec3_in_sphere())

# ╔═╡ 485f9c5b-4c5d-453c-b190-e84ae0cd1a21
# """Create a scattered ray emitted by `mat` from incident Ray `r`. 

# 	Args:
# 		rec: the HitRecord of the surface from which to scatter the ray.

# 	Return missing if it's fully absorbed. """
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

# ╔═╡ c299ca47-46c4-42da-9f8b-ae3abbeb6e51
function scatter(mat::Metal, r_in::Ray, rec::HitRecord)::Scatter
	reflected = normalize(reflect(r_in.dir, rec.n⃗) + mat.fuzz*random_vec3_on_sphere())
	Scatter(Ray(rec.p, reflected), mat.albedo)
end

# ╔═╡ f72214f9-03c4-4ba3-bb84-069256446b31
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

# ╔═╡ 64104df6-4b79-4329-bfed-14619aa73e3c
"""Render an image of `scene` using the specified camera, number of samples.

	Args:
		scene: a HittableList, e.g. a list of spheres
		n_samples: number of samples per pixel, eq. to C++ samples_per_pixel

	Equivalent to C++'s `main` function."""
function render(scene::HittableList, cam::Camera, image_width=400,
				n_samples=1)
	# Image
	aspect_ratio = 16.0/9.0
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

# ╔═╡ aa38117f-45e8-4070-a412-958f0ce19aa5
render(scene_2_spheres(), default_camera(), 96, 16)

# ╔═╡ 9fd417cc-afa9-4f12-9c29-748f0522554c
render(scene_4_spheres(), default_camera(), 96, 16)

# ╔═╡ a2221922-31be-42f3-8f70-845fae385d2c
render(scene_4_spheres(), default_camera(), 200, 32)

# ╔═╡ a1564d79-3628-4121-99a9-d3674e16eb04
render(scene_diel_spheres(), default_camera(), 96, 16)

# ╔═╡ 9cad61ba-6b12-4681-b927-2689b12e9a0d
random_vec3_on_sphere()

# ╔═╡ fbd3a778-60e5-412a-869f-f2a15f605c34
norm(random_vec3_on_sphere())

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[compat]
Images = "~0.25.0"
StaticArrays = "~1.2.13"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "e527b258413e0c6d4f66ade574744c94edef81f8"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.40"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f885e7e7c124f8c92650d61b9477b9ac2ee607dd"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.1"

[[ChangesOfVariables]]
deps = ["LinearAlgebra", "Test"]
git-tree-sha1 = "9a1d594397670492219635b35a3d830b04730d62"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.1"

[[Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "75479b7df4167267d75294d14b58244695beb2ac"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.2"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "3f1f500312161f1ae067abe07d13b40f78f32e07"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.8"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dce3e3fea680869eaa0b774b2e8343e9ff442313"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.40.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "681ea870b918e7cff7111da58791d7f718067a19"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.2"

[[CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[DataStructures]]
deps = ["InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "88d48e133e6d3dd68183309877eac74393daa7eb"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.17.20"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "837c83e5574582e07662bbbba733964ff7c26b9d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.6"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "9aad812fb7c4c038da7cab5a069f502e6e3ae030"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.1"

[[FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "463cb335fa22c4ebacfd1faba5fde14edb80d96c"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.5"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2db648b6712831ecb333eae76dbfd1c156ca13bb"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.11.2"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "1c5a84319923bea76fa145d49e93aa4394c73fc2"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.1"

[[Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "92243c07e786ea3458532e199eb3feee0e7e08eb"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.4.1"

[[IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "c54b581a83008dc7f292e205f4c409ab5caa0f04"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.10"

[[ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "b51bb8cae22c66d0f6357e3bcb6363145ef20835"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.5"

[[ImageContrastAdjustment]]
deps = ["ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "0d75cafa80cf22026cea21a8e6cf965295003edc"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.10"

[[ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "9a5c62f231e5bba35695a20988fc7cd6de7eeb5a"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.9.3"

[[ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "7a20463713d239a19cbad3f6991e404aca876bda"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.15"

[[ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "15bd05c1c0d5dbb32a9a3d7e0ad2d50dd6167189"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.1"

[[ImageIO]]
deps = ["FileIO", "Netpbm", "OpenEXR", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "a2951c93684551467265e0e32b577914f69532be"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.9"

[[ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[ImageMagick_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "ea2b6fd947cdfc43c6b8c15cff982533ec1f72cd"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.12+0"

[[ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "36cbaebed194b292590cba2593da27b34763804a"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.8"

[[ImageMorphology]]
deps = ["ImageCore", "LinearAlgebra", "Requires", "TiledIteration"]
git-tree-sha1 = "5581e18a74a5838bd919294a7138c2663d065238"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.3.0"

[[ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "OffsetArrays", "Statistics"]
git-tree-sha1 = "1d2d73b14198d10f7f12bf7f8481fd4b3ff5cd61"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.0"

[[ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "36832067ea220818d105d718527d6ed02385bf22"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.7.0"

[[ImageShow]]
deps = ["Base64", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "d0ac64c9bee0aed6fdbb2bc0e5dfa9a3a78e3acc"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.3"

[[ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "b4b161abc8252d68b13c5cc4a5f2ba711b61fec5"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.9.3"

[[Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "35dc1cd115c57ad705c7db9f6ef5cc14412e8f00"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.25.0"

[[Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "87f7662e03a649cffa2e05bf19c303e168732d3e"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.2+0"

[[IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IntegralArrays]]
deps = ["IntervalSets"]
git-tree-sha1 = "26c4df96fdf6127a92d53cb8ffb577104617adca"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "61aa005707ea2cebf47c8d780da8dc9bc4e0c512"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.4"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[JLD2]]
deps = ["DataStructures", "FileIO", "MacroTools", "Mmap", "Pkg", "Printf", "Reexport", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "46b7834ec8165c541b0b5d1c8ba63ec940723ffb"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.15"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "be9eef9f9d78cecb6f262f3c10da151a6c5ab827"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "5455aef09b40e5020e1520f551fa3135040d4ed0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+2"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[MappedArrays]]
git-tree-sha1 = "e8b359ef06ec72e8c030463fe02efe5527ee5142"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.1"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "2af69ff3c024d13bde52b34a2a7d6887d4e7b438"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.7.1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "16baacfdc8758bc374882566c9187e785e85c2f0"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.9"

[[Netpbm]]
deps = ["FileIO", "ImageCore"]
git-tree-sha1 = "18efc06f6ec36a8b801b23f076e3c6ac7c3bf153"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.2"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "923319661e9a22712f24596ce81c54fc0366f304"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.1+0"

[[OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "6d105d40e30b635cfed9d52ec29cf456e27d38f8"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.12"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "646eed6f6a5d8df6708f15ea7e02a7a2c4fe4800"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.10"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[Ratios]]
deps = ["Requires"]
git-tree-sha1 = "01d341f502250e81f6fec0afe662aa861392a3aa"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.2"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rotations]]
deps = ["LinearAlgebra", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "6a23472b6b097d66da87785b61137142ac104f94"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.0.4"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "a6f404cc44d3d3b28c793ec0eb59af709d827e4e"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.2.1"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e08890d19787ec25029113e88c34ec20cac1c91e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.0.0"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "e7bc80dc93f50857a5d1e3c8121495852f407e6a"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3c76dde64d03699e074ac02eb2e8ba8254d428da"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.13"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "0f2aa8e32d511f758a2ce49208181f7733a0936a"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.1.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2bb0cb32026a66037360606510fca5984ccc6b75"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.13"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TiffImages]]
deps = ["ColorTypes", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "OrderedCollections", "PkgVersion", "ProgressMeter"]
git-tree-sha1 = "945b8d87c5e8d5e34e6207ee15edb9d11ae44716"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.4.3"

[[TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "5683455224ba92ef59db72d10690690f4a8dc297"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.1"

[[TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "de67fa59e33ad156a590055375a30b23c40299d3"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─0866add2-9b95-45e7-8081-c01cd2a66911
# ╟─611d5eae-4b09-11ec-27bf-ef4a1ecdcc41
# ╟─97bb4432-ed41-423b-b4d9-bafc519de641
# ╠═38fdd4ef-c383-4f97-8451-c6f602307e7d
# ╠═84d29423-cf11-41c3-af4d-c5f63b1ef23e
# ╠═7d26fef0-9a06-479a-ae08-e9d04e455767
# ╠═f29ad2c0-c3ff-484d-8fdd-dff34d2bb863
# ╠═2192e695-4378-4b47-8ce0-353636cd2cd1
# ╠═538d1aa5-07f9-4fca-8410-ef63b8a6857b
# ╠═8aeb7373-6bb0-4544-8655-fa941561688c
# ╠═154d736b-8fdc-44af-ae2a-9e5ba6d2c92e
# ╟─216922d8-613a-4ac1-9559-40878e6587e2
# ╠═961fd749-d439-4dfa-ae21-b1659dc54511
# ╟─7249557b-043c-4994-bec7-615f137f98e3
# ╠═3dceca5d-7d1e-425b-9516-24e0a24adaff
# ╠═f8007c75-9487-414a-9592-138a696c2957
# ╠═668030c8-24a7-4aa6-b858-cedf8ac5f988
# ╠═3eb50f44-9091-45e8-a7e1-92d25b4b2090
# ╠═78f209df-d176-4711-80fc-a8054771f105
# ╠═e88de775-6904-4182-8209-06db22758470
# ╠═3e6fd5c0-6d4a-44ef-a7b2-106b52fc6550
# ╠═5fd1ec87-3616-448a-ab4d-fede804b26d5
# ╠═a0893bf4-9607-4853-8162-9f34d3337060
# ╠═cfbcb883-d12e-4ad3-a084-064749bddcdb
# ╠═6348c03d-e8ec-4dbb-9f8a-8e4a48bb1cb3
# ╠═dbc8fc2d-39c2-4ec9-b82d-7c6a8b12dde7
# ╠═53832af1-a9be-4e02-8b71-a70dae63c233
# ╠═81b4c9e4-9f93-45ca-9fa0-7e9686a55e9a
# ╟─678214c5-de81-489f-b002-c343d48071c9
# ╠═cbb6418c-79e9-4359-80a6-40a8fa40679e
# ╠═14be6068-6a15-4fad-ac0a-f156da71f103
# ╠═64ef0313-2d2b-49d5-a1a1-3b04426a82f8
# ╠═971777a6-f269-4344-8dba-7a55118396e5
# ╠═655ffa6c-f1e9-4149-8f1d-51145c5a51e4
# ╟─9075f8ed-f319-486d-94b2-486806aba3fd
# ╠═c00e2004-2002-4dd2-98ed-f898ef2c14f1
# ╠═b7399fb8-6205-41ea-9c70-eb62daedcefb
# ╠═1d04159d-87bd-4cf8-a73c-817f20ca1026
# ╟─fed81f09-e4a0-433f-99e8-261096114b7b
# ╠═24e8740a-8e44-4206-b2b6-c4a55002dad8
# ╠═359832af-7598-4c45-8033-c28cb0d86772
# ╠═ed6ab8be-587c-4cb6-8172-618c74d3f9cc
# ╠═a65c68c9-e489-465a-9687-93ae9da14a5e
# ╠═2c4b4453-1a46-4889-9a14-16b18cc8c240
# ╠═98c43f3f-4bfc-49db-806a-850b7d75b5a4
# ╠═3b570d37-f407-41d8-b8a0-a0af4d85b14d
# ╠═138bb5b6-0f45-4f13-8339-5110eb7cd1ff
# ╟─6b36d245-bf01-45a7-b119-8315226dd4a3
# ╠═4a396b3f-f920-4ec2-91f6-7d61fe2b9699
# ╠═427f247c-055c-459e-9862-26e9f6f3e24f
# ╠═88e51c27-0f28-4dcc-b9e9-ac44eeb876f5
# ╠═7c4a67b2-8208-4cd4-b1ea-16f6f50adfe8
# ╠═ca649864-5a6d-4ca7-896e-e80e8a48443e
# ╠═485f9c5b-4c5d-453c-b190-e84ae0cd1a21
# ╠═78efebc5-53fd-417d-bd9e-667fd504e3fd
# ╠═05e57afd-6eb9-42c5-9666-7be3771fa6b8
# ╠═08e18ae5-9927-485e-9644-552f03e06f27
# ╠═737e2f87-82f5-45b6-a76c-4f560c29f5b9
# ╠═f72214f9-03c4-4ba3-bb84-069256446b31
# ╠═0bf88264-d4c5-4d5a-babe-d2433e46024d
# ╠═555bea1d-5178-48dd-87e6-4e2a2471a5dd
# ╠═c299ca47-46c4-42da-9f8b-ae3abbeb6e51
# ╟─851c002c-dc23-4999-b28c-a716c5d2d42c
# ╠═70530f8e-1b29-4588-927f-d38d5d12d5c9
# ╠═c5349670-4df4-421f-9d5a-b28c1b9040c2
# ╟─282a4912-7a6e-44ae-90eb-f2f7c8f3d0f4
# ╠═a0e5a1f3-244f-427b-a335-7e233af1d9d8
# ╠═5d00f26b-35f2-4071-8e04-227ffc25f184
# ╠═c1aef1be-79d4-4417-be36-ae8416465986
# ╠═94081092-afc6-4359-bd2c-15e8407bf70e
# ╠═813eaa13-2eb2-4302-9e4d-5d1dab0ac7c4
# ╠═a4493f3a-cb9f-404e-830d-a7d007df5baf
# ╟─891ce2c8-f8b2-472b-a8d9-389dafddcf22
# ╠═64104df6-4b79-4329-bfed-14619aa73e3c
# ╟─97d9a286-2e70-4dd4-8407-62b3a89da16b
# ╠═aa38117f-45e8-4070-a412-958f0ce19aa5
# ╠═9fd417cc-afa9-4f12-9c29-748f0522554c
# ╠═4dd59aa7-37a7-426b-8573-a0fee26343df
# ╠═a2221922-31be-42f3-8f70-845fae385d2c
# ╟─5f1bae02-d425-4a73-8668-d6383faba79d
# ╠═776b5096-4792-46b9-b70f-50a2b4c4999b
# ╠═9dc64353-c41c-45b4-aacd-12d5d6117c58
# ╠═e1adf736-592e-4980-9318-a641a2188b8a
# ╠═bbfd4db5-3650-4f27-9777-2ff981c3d28b
# ╠═b54e631c-12ab-49c1-a63e-65b1c9a5d8b6
# ╠═f5c4e502-048c-4fcd-879f-eaeb4430c012
# ╠═ae3b8f15-985d-4f74-ac8c-86a3ffc3b8b1
# ╠═ddf5883c-036a-4a21-908d-bb7cec731f7f
# ╠═a1564d79-3628-4121-99a9-d3674e16eb04
# ╟─30102751-fbbd-41dc-9dc1-5c7cb8cd613f
# ╠═1f4a9699-5c91-4e2c-b592-1bfa86c05959
# ╠═795fdf6f-945e-44f8-8aa1-1e33586cc095
# ╠═f7d47388-076b-416c-a088-c8963009faa6
# ╠═927e351c-3529-4dc1-b342-e5c040fdc6e2
# ╠═7e98cea8-90b4-44da-b234-a60c1a3b2fa4
# ╠═58e91524-be98-4b67-bd9a-5c682cb5c009
# ╠═44fc6b51-c682-4fa9-9ebe-804a4f3397be
# ╠═a0c4f8dc-5580-413b-99ee-dcb7b6c92c8d
# ╠═9cad61ba-6b12-4681-b927-2689b12e9a0d
# ╠═fbd3a778-60e5-412a-869f-f2a15f605c34
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
