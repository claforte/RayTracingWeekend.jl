module RayTracingWeekend

using Images, LinearAlgebra, Random, RandomNumbers.Xorshifts, StaticArrays

export color_vec3_in_rgb, default_camera, get_ray, hit, near_zero, point, random_between, random_vec2, 
        random_vec2_in_disk, random_vec3, random_vec3_in_sphere, random_vec3_on_sphere, ray_color, ray_to_HitRecord, reflect, 
        reflectance, refract, render, reseed!, rgb, rgb_gamma2, skycolor, squared_length, trand
export Camera, Dielectric, Hittable, HittableList, HitRecord, Lambertian, Material, Metal, Ray, Scatter, Sphere, Vec3
export scene_2_spheres, scene_4_spheres, scene_blue_red_spheres, scene_diel_spheres, scene_random_spheres

const Vec3{T<:AbstractFloat} = SVector{3, T}

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
@inline near_zero(v::Vec3) = squared_length(v) < 1e-5
@inline rgb(v::Vec3) = RGB(v...)
@inline rgb_gamma2(v::Vec3) = RGB(sqrt.(v)...)

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

# Instantiate 1 RNG (Random Number Generator) per thread, for performance
# Fix the random seeds, to make it easier to benchmark changes.
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

@inline function random_vec3_in_sphere(::Type{T}) where T # equiv to random_in_unit_sphere()
	while true
		p = random_vec3(T(-1), T(1))
		if p⋅p <= 1
			return p
		end
	end
end

@inline random_between(min::T=0, max::T=1) where T = trand(T)*(max-min) + min # equiv to random_double()
@inline random_vec3(min::T, max::T) where T = @inbounds @SVector[random_between(min, max) for i ∈ 1:3]
@inline random_vec2(min::T, max::T) where T = @inbounds @SVector[random_between(min, max) for i ∈ 1:2]

"Random unit vector. Equivalent to C++'s `unit_vector(random_in_unit_sphere())`"
@inline random_vec3_on_sphere(::Type{T}) where T = normalize(random_vec3_in_sphere(T))

@inline function random_vec2_in_disk(::Type{T}) where T # equiv to random_in_unit_disk()
	while true
		p = random_vec2(T(-1), T(1))
		if p⋅p <= 1
			return p
		end
	end
end

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

struct Lambertian{T} <: Material{T}
	albedo::Vec3{T}
end

"""Compute reflection vector for v (pointing to surface) and normal n⃗.

	See [diagram](https://raytracing.github.io/books/RayTracingInOneWeekend.html#metal/mirroredlightreflection)"""
@inline @fastmath reflect(v::Vec3{T}, n⃗::Vec3{T}) where T = v - (2v⋅n⃗)*n⃗

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

struct Metal{T} <: Material{T}
	albedo::Vec3{T}
	fuzz::T # how big the sphere used to generate fuzzy reflection rays. 0=none
	@inline Metal(a::Vec3{T}, f::T=0.0) where T = new{T}(a,f)
end
@inline @fastmath function scatter(mat::Metal{T}, r_in::Ray{T}, rec::HitRecord)::Scatter{T} where T
	reflected = normalize(reflect(r_in.dir, rec.n⃗) + mat.fuzz*random_vec3_on_sphere(T))
	Scatter(Ray(rec.p, reflected), mat.albedo)
end

"Scene with 2 Lambertian spheres"
function scene_2_spheres(; elem_type::Type{T}) where T
	spheres = Sphere[]
	
	# small center sphere
	push!(spheres, Sphere((SA{T}[0,0,-1]), T(0.5), Lambertian(SA{T}[0.7,0.3,0.3])))
	
	# ground sphere
	push!(spheres, Sphere((SA{T}[0,-100.5,-1]), T(100), Lambertian(SA{T}[0.8,0.8,0.0])))
	HittableList(spheres)
end

"""Scene with 2 Lambertian, 2 Metal spheres.

	See https://raytracing.github.io/images/img-1.11-metal-shiny.png"""
function scene_4_spheres(; elem_type::Type{T}) where T
	scene = scene_2_spheres(; elem_type=elem_type)

	# left and right Metal spheres
	push!(scene, Sphere((SA{T}[-1,0,-1]), T(0.5), Metal((SA{T}[0.8,0.8,0.8]), T(0.3)))) 
	push!(scene, Sphere((SA{T}[ 1,0,-1]), T(0.5), Metal((SA{T}[0.8,0.6,0.2]), T(0.8))))
	return scene
end

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

@inline @fastmath function get_ray(c::Camera{T}, s::T, t::T) where T
	rd = SVector{2,T}(c.lens_radius * random_vec2_in_disk(T))
	offset = c.u * rd.x + c.v * rd.y #offset = c.u * rd.x + c.v * rd.y
    Ray(c.origin + offset, normalize(c.lower_left_corner + s*c.horizontal +
									 t*c.vertical - c.origin - offset))
end

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

	#Threads.@threads # claforte: uncomment for CRASH?!
	for i in 1:image_height
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

function scene_blue_red_spheres(; elem_type::Type{T}) where T # dielectric spheres
	spheres = Sphere[]
	R = cos(pi/4)
	push!(spheres, Sphere((SA{T}[-R,0,-1]), R, Lambertian(SA{T}[0,0,1]))) 
	push!(spheres, Sphere((SA{T}[ R,0,-1]), R, Lambertian(SA{T}[1,0,0]))) 
	HittableList(spheres)
end

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

end
