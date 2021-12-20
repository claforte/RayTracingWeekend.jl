@inline function skycolor(ray::Ray{T}) where T
	white = SA[1.0, 1.0, 1.0]
	skyblue = SA[0.5, 0.7, 1.0]
	t = T(0.5)*(ray.dir.y + one(T))
    (one(T)-t)*white + t*skyblue
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
