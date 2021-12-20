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

default_camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, focus_dist; elem_type::Type{T}) where T =
	default_camera(Vec3{T}(lookfrom), Vec3{T}(lookat), Vec3{T}(vup), 
		T(vfov), T(aspect_ratio), T(aperture), T(focus_dist)
	)

@inline @fastmath function get_ray(c::Camera{T}, s::T, t::T) where T
	rd = SVector{2,T}(c.lens_radius * random_vec2_in_disk(T))
	offset = c.u * rd.x + c.v * rd.y #offset = c.u * rd.x + c.v * rd.y
    Ray(c.origin + offset, normalize(c.lower_left_corner + s*c.horizontal +
									 t*c.vertical - c.origin - offset))
end
