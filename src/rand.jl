# Reset the per-thread random seeds to make results reproducible
reseed!() = for i in 1:Threads.nthreads() Random.seed!(TRNG[i], i) end

"Per-thread rand()"
@inline function trand() 
	@inbounds rng = TRNG[Threads.threadid()]
	rand(rng)
end

@inline function trand(::Type{T}) where T 
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