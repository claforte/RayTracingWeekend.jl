# Concrete materials

struct Lambertian{T} <: Material{T}
	albedo::Vec3{T}
end

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

struct Metal{T} <: Material{T}
	albedo::Vec3{T}
	fuzz::T # how big the sphere used to generate fuzzy reflection rays. 0=none
	@inline Metal(a::Vec3{T}, f::T=0.0) where T = new{T}(a,f)
end

@inline @fastmath function scatter(mat::Metal{T}, r_in::Ray{T}, rec::HitRecord)::Scatter{T} where T
	reflected = normalize(reflect(r_in.dir, rec.n⃗) + mat.fuzz*random_vec3_on_sphere(T))
	Scatter(Ray(rec.p, reflected), mat.albedo)
end

"Dielectric, i.e. transparent/refractive material"
struct Dielectric{T} <: Material{T}
	ir::T # index of refraction, i.e. η.
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
