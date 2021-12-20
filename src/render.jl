"""Render an image of `scene` using the specified camera, number of samples.

	Args:
		scene: a HittableList, e.g. a list of spheres
		n_samples: number of samples per pixel, eq. to C++ samples_per_pixel

	Equivalent to C++'s `main` function."""
function render(scene::HittableList, cam::Camera{T}, image_width=400,
				n_samples=1) where T
	# Image
	aspect_ratio = 16//9 # TODO: use cam.aspect_ratio for consistency
	image_height = image_width ÷ aspect_ratio

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
