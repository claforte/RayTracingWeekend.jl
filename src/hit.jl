# Calculate intersections with surface(s)

@inline point(r::Ray{T}, t::T) where T <: AbstractFloat = r.origin .+ t .* r.dir # equivalent to C++'s ray.at()

"""Equivalent to `hit_record.set_face_normal()`"""
@inline @fastmath function ray_to_HitRecord(t::T, p, outward_n⃗, r_dir::Vec3{T}, mat::Material{T})::Union{HitRecord,Bool} where T
	front_face = r_dir ⋅ outward_n⃗ < 0
	n⃗ = front_face ? outward_n⃗ : -outward_n⃗
	HitRecord(t,p,n⃗,front_face,mat)
end


"""
    Return: one of:
        HitRecord: if a hit occured with actual geometry
        true: if a hit occured, e.g. with a boundingbox, when we don't care about the exact location
        false: if not hit occured """
@inline @fastmath function hit(s::Sphere{T}, r::Ray{T}, tmin::T, tmax::T)::Union{HitRecord,Bool} where T
    oc = r.origin - s.center
    #a = r.dir ⋅ r.dir # unnecessary since `r.dir` is normalized
	a = 1
	half_b = oc ⋅ r.dir
    c = oc⋅oc - s.radius^2
    discriminant = half_b^2 - a*c
	if discriminant < 0 return false end
	sqrtd = √discriminant
	
	# Find the nearest root that lies in the acceptable range
	root = (-half_b - sqrtd) / a	
	if root < tmin || tmax < root
		root = (-half_b + sqrtd) / a
		if root < tmin || tmax < root
			return false
		end
	end
	
	t = root
	p = point(r, t)
	n⃗ = (p - s.center) / s.radius
	return ray_to_HitRecord(t, p, n⃗, r.dir, s.mat)
end

@inline @fastmath function hit(ms::MovingSphere{T}, r::Ray{T}, tmin::T, tmax::T)::Union{HitRecord,Bool} where T
    hit(sphere(ms, r.time), r, tmin, tmax)
end

"""Find closest hit between `Ray r` and a list of Hittable objects `h`, within distance `tmin` < `tmax`"""
@inline function hit(hittables::HittableList, r::Ray{T}, tmin::T, tmax::T)::Union{HitRecord,Bool} where T
    closest = tmax # closest t so far
    best_rec::Union{HitRecord,Bool} = false # by default, no hit
    @inbounds for i in eachindex(hittables)
		h = hittables[i]
        rec = hit(h, r, tmin, closest)
        if typeof(rec) == HitRecord
            best_rec = rec
            closest = best_rec.t # i.e. ignore any further hit > this one's.
        end
    end
    best_rec
end

