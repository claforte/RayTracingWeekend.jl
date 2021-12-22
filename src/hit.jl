# Calculate intersections with surface(s)

@inline point(r::Ray{T}, t::T) where T <: AbstractFloat = r.origin .+ t .* r.dir # equivalent to C++'s ray.at()

# Bounding boxes
#---------------

"Compute bounding box for the sphere. Equivalent to C++ bounding_box"
@inline function bbox(sphere::Sphere{T}, time0::T, time1::T)::Aabb where T
	rad = sphere.radius
	Aabb(sphere.center - SA[rad,rad,rad], sphere.center + SA[rad,rad,rad])
end

"Compute the bounding box of 2 AABBs"
@inline function bbox(b1::Aabb, b2::Aabb)::Aabb where T
	Aabb(SA[min(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])],
		 SA[max(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])])
end

@inline function bbox(ms::MovingSphere{T}, time0::T, time1::T)::Aabb where T
	rad = ms.radius
	b1 = bbox(sphere(ms, ms.time0), t0, t1)
	b2 = bbox(sphere(ms, ms.time1), t0, t1)
	bbox(b1, b2)
end

@inline function bbox(hittables::HittableList, time0::T, time1::T)::Aabb where T
	bb = hittables[1]
	@inbounds for i in eachindex(hittables)
		bb = bbox(bb, hittables[i])
	end
	bb
end


# HitRecord/Hit
# -------------

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

# Adapted from Andrew Kensler's version
@inline @fastmath function hit(box::Aabb{T}, r::Ray{T}, tmin::T, tmax::T)::Union{HitRecord,Bool} where T
	for a in 1:3 # axis   # @inbounds? @simd? @turbo?
		invD = T(1.0) / r.dir[a]
		t0 = (box.min[a] - r.origin[a]) * invD
		t1 = (box.max[a] - r.origin[a]) * invD
		if invD < T(0)
			(t0,t1) = t1,t0 # swap
		end
		tmin = t0 > tmin ? t0 : tmin 
		tmax = t1 < tmax ? t1 : tmax
		if (tmax <= tmin)
			println("a=$a")
			return false # no hit!
		end
	end

	return true
end

@inline function hit(bvh::BvhNode, r::Ray{T}, tmin::T, tmax::T)::Union{HitRecord,Bool} where T
    if !hit(bvh_node.box, r, tmin, tmax) return false end

	hit_left = hit(bvh.left, r, tmin, tmax) # HitRecord or false...
	hit_right = hit(bvh.right, r, tmin, hit_left !== false ? hit_left.t : tmax)

	if hit_right !== false
		return hit_right # automatically takes into consideration whether hit_left occured
	else
		return hit_left
	end
end
