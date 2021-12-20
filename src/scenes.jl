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
