using RayTracingWeekend
using Test
using BenchmarkTools, Images, InteractiveUtils, LinearAlgebra, StaticArrays

@testset "Dep_Images" begin # dependencies on Images.jl
    img = rand(4, 3)
    img_rgb = rand(RGB, 4, 4)

    ex1 = [RGB(1,0,0) RGB(0,1,0) RGB(0,0,1);
        RGB(1,1,0) RGB(1,1,1) RGB(0,0,0)]

    ex2 = zeros(RGB, 2, 3)
    ex2[1,1] = RGB(1,0,0)
    ex2

    # The "Hello World" of graphics
    @fastmath function gradient(nx::Int, ny::Int)
        img = zeros(RGB, ny, nx)
        @inbounds for j in 1:nx
            for i in 1:ny # Julia is column-major, i.e. iterate 1 column at a time
                x = j; y = (ny-i);
                r = x/nx
                g = y/ny
                b = 0.2
                img[i,j] = RGB(r,g,b)
            end
        end
        img
    end

    gradient(20,10)
end

@testset "RayTracingWeekend.jl" begin
    t_col = SA[0.4, 0.5, 0.1] # test color
    @test (@ballocated squared_length($t_col)) == 0 # 1.162 ns

    @test !near_zero(t_col)
    @test (@ballocated near_zero($t_col)) == 0 # 1.382 ns


    @show rgb(t_col)
    # #rgb($t_col) # 1.172 ns (0 allocations: 0 bytes)

    # #rgb_gamma2($t_col) # 3.927 ns (0 allocations: 0 bytes)

    # _origin = SA[0.0,0.0,0.0]
    # _v3_minusY = SA[0.0,-1.0,0.0]
    # _t_ray1 = Ray(_origin, _v3_minusY)
    # # Float32: 6.161 ns (0 allocations: 0 bytes)
    # # Float64: 6.900 ns (0 allocations: 0 bytes)
    # #@btime Ray($_origin, $_v3_minusY) 

    # #@btime point($_t_ray1, 0.5) # 1.412 ns (0 allocations: 0 bytes)

    # # 1.402 ns (0 allocations: 0 bytes)
    # #@btime skycolor($_t_ray1)

    # # 1.412 ns (0 allocations: 0 bytes)
    # #@btime rgb(skycolor($_t_ray1)) # 291.492 ns (4 allocations: 80 bytes)

    # #@btime trand()

    # #    2.695 ns (0 allocations: 0 bytes)
    # #@btime random_between(50.0, 100.0) 

    # #@btime random_vec3(-1.0,1.0)

    # #@btime random_vec2(-1.0f0,1.0f0) # 3.677 ns (0 allocations: 0 bytes)

    # # REMEMBER: the times are somewhat random! Use best timing of 5!
    # # Float32: 34.690 ns (0 allocations: 0 bytes)
    # # Float64: 34.065 ns (0 allocations: 0 bytes)
    # # rand() using MersenneTwister _rng w/ Float64:
    # #   21.333 ns (0 allocations: 0 bytes)
    # # rand() using Xoroshiro128Plus _rng w/ Float64:
    # #   19.716 ns (0 allocations: 0 bytes)
    # random_vec3_in_sphere(Float64)

    # "Random unit vector. Equivalent to C++'s `unit_vector(random_in_unit_sphere())`"
    # @inline random_vec3_on_sphere(::Type{T}) where T = normalize(random_vec3_in_sphere(T))

    # # REMEMBER: the times are somewhat random! Use best timing of 5! 
    # # Before optim:
    # #   517.538 ns (12 allocations: 418 bytes)... but random
    # # After reusing _rand_vec3:
    # #    92.865 ns (2 allocations: 60 bytes)
    # # Various speed-ups (don't use Vec3, etc.): 
    # #    52.427 ns (0 allocations: 0 bytes)
    # # Float64: 34.549 ns (0 allocations: 0 bytes)
    # # Float32: 36.274 ns (0 allocations: 0 bytes)
    # # rand() using Xoroshiro128Plus _rng w/ Float64:
    # #   19.937 ns (0 allocations: 0 bytes)
    # random_vec3_on_sphere(Float32)

    # @inline function random_vec2_in_disk(::Type{T}) where T # equiv to random_in_unit_disk()
    #     while true
    #         p = random_vec2(T(-1), T(1))
    #         if p⋅p <= 1
    #             return p
    #         end
    #     end
    # end

    # # Keep best timing of 5!
    # # Float32: 14.391 ns (0 allocations: 0 bytes)
    # # Float64: 14.392 ns (0 allocations: 0 bytes)
    # # rand() using MersenneTwister _rng w/ Float64:
    # #   7.925 ns (0 allocations: 0 bytes)
    # # rand() using Xoroshiro128Plus _rng w/ Float64:
    # #   7.574 ns (0 allocations: 0 bytes)
    # random_vec2_in_disk(Float64)

    # @inline @fastmath function hit_sphere1(center::Vec3{T}, radius::T, r::Ray{T}) where T
    #     oc = r.origin - center
    #     #a = r.dir ⋅ r.dir # unnecessary since `r.dir` is normalized
    #     a = 1
    #     b = 2oc ⋅ r.dir
    #     c = (oc ⋅ oc) - radius*radius
    #     discriminant = b*b - 4a*c
    #     discriminant > 0
    # end
    
    # @inline function sphere_scene1(r::Ray{T}) where T
    #     if hit_sphere1(SA[T(0), T(0), T(-1)], T(0.5), r) # sphere of radius 0.5 centered at z=-1
    #         return SA[T(1),T(0),T(0)] # red
    #     else
    #         skycolor(r)
    #     end
    # end
    
    


    # """ Temporary function to shoot rays through each pixel. Later replaced by `render`
        
    #     Args:
    #         scene: a function that takes a ray, returns the color of any object it hit
    # """
    # function main(nx::Int, ny::Int, scene, ::Type{T}) where T
    #     lower_left_corner = SA[-2,-1,-1]
    #     horizontal = SA[4,0,0]
    #     vertical = SA[T(0),T(2),T(0)]
    #     origin = SA[T(0),T(0),T(0)]
        
    #     img = zeros(RGB{T}, ny, nx)
    #     Threads.@threads for j in 1:nx
    #         @inbounds for i in 1:ny # Julia is column-major, i.e. iterate 1 column at a time
    #             u = T(j/nx)
    #             v = T((ny-i)/ny) # Y-up!
    #             ray = Ray(origin, normalize(lower_left_corner + u*horizontal + v*vertical))
    #             #r = x/nx
    #             #g = y/ny
    #             #b = 0.2
    #             img[i,j] = rgb(scene(ray))
    #         end
    #     end
    #     img
    # end

    # # Before optimizations:
    # #   28.630 ms (520010 allocations: 13.05 MiB)
    # # After optimizations (skycolor, rand, etc.)
    # #   10.358 ms (220010 allocations: 5.42 MiB) # at this stage, rgb() is most likely the culprit..
    # # After replacing Color,Vec3 by @SVector:
    # #    6.539 ms (80002 allocations: 1.75 MiB)
    # # Lots more optimizations, including @inline of low-level functions:
    # #  161.546 μs (     2 allocations: 234.45 KiB)
    # # With Vec3{T}...
    # # Float32: 174.560 μs (2 allocations: 234.45 KiB)
    # # Float64: 161.516 μs (2 allocations: 468.83 KiB)
    # # Above was all using 1 single thread. With 16 threads:
    # #   52.218 μs (83 allocations: 241.75 KiB)
    # # With @inbounds:
    # #   49.243 μs (83 allocations: 241.75 KiB)
    # main(200, 100, skycolor, Float32) 

    # #md"# Chapter 5: Add a sphere"

    # # @fastmath speeds up a lot!
    # @inline @fastmath function hit_sphere1(center::Vec3{T}, radius::T, r::Ray{T}) where T
    #     oc = r.origin - center
    #     #a = r.dir ⋅ r.dir # unnecessary since `r.dir` is normalized
    #     a = 1
    #     b = 2oc ⋅ r.dir
    #     c = (oc ⋅ oc) - radius*radius
    #     discriminant = b*b - 4a*c
    #     discriminant > 0
    # end

    # @inline function sphere_scene1(r::Ray{T}) where T
    #     if hit_sphere1(SA[T(0), T(0), T(-1)], T(0.5), r) # sphere of radius 0.5 centered at z=-1
    #         return SA[T(1),T(0),T(0)] # red
    #     else
    #         skycolor(r)
    #     end
    # end

    # # Float32: 149.613 μs (2 allocations: 234.45 KiB)
    # # Float64: 173.818 μs (2 allocations: 468.83 KiB)
    # # w/ @fastmath in hit_sphere1:
    # #    63.751 μs (83 allocations: 476.14 KiB)
    # # Eliminate unnecessary r.dir ⋅ r.dir: 
    # #    61.416 μs (83 allocations: 476.14 KiB)
    # main(200, 100, sphere_scene1, Float64) 

    # #md"# Chapter 6: Surface normals and multiple objects"

    # @inline function hit_sphere2(center::Vec3{T}, radius::T, r::Ray{T}) where T
    #     oc = r.origin - center
    #     #a = r.dir ⋅ r.dir # unnecessary since `r.dir` is normalized
    #     a = 1
    #     b = 2oc ⋅ r.dir
    #     c = (oc ⋅ oc) - radius^2
    #     discriminant = b*b - 4a*c
    #     if discriminant < 0
    #         return -1
    #     else
    #         return (-b - sqrt(discriminant)) / 2a
    #     end
    # end

    # @inline function sphere_scene2(r::Ray{T}) where T
    #     sphere_center = SA[T(0),T(0),T(-1)]
    #     t = hit_sphere2(sphere_center, T(0.5), r) # sphere of radius 0.5 centered at z=-1
    #     if t > T(0)
    #         n⃗ = normalize(point(r, t) - sphere_center) # normal vector. typed n\vec
    #         return T(0.5)*n⃗ + SA[T(0.5),T(0.5),T(0.5)] # remap normal to rgb
    #     else
    #         skycolor(r)
    #     end
    # end

    # # claforte: this got significantly worse after Vec3{T} was integrated...
    # # WAS: 
    # # Float32: 262.286 μs (2 allocations: 234.45 KiB)
    # # NOW:
    # # Float32: 290.470 μs (2 allocations: 234.45 KiB)
    # # Float64: 351.164 μs (2 allocations: 468.83 KiB)
    # # Above was all using 1 single thread. With 16 threads:
    # #   64.612 μs (83 allocations: 241.75 KiB)
    # # Eliminate unnecessary r.dir ⋅ r.dir: 
    # #   61.586 μs (82 allocations: 241.72 KiB)
    # #print("@btime main(200,100,sphere_scene2, Float32):")
    # main(200,100,sphere_scene2, Float32)

    # @assert reflect(SA[0.6,-0.8,0.0], SA[0.0,1.0,0.0]) == SA[0.6,0.8,0.0] # diagram's example

    # default_camera(SA[0f0,0f0,0f0]) # Float32 camera

    # get_ray(default_camera(SA[0f0,0f0,0f0]), 0.0f0, 0.0f0)
    # #@btime get_ray(default_camera(), 0.0f0, 0.0f0)

    # # Float32/Float64
    # ELEM_TYPE = Float64

    # t_default_cam = default_camera(SA{ELEM_TYPE}[0,0,0])

    # # Using @paulmelis' style of hit(): @inbounds for i in eachindex(hittables) and Union{HitRecord, Nothing}
    # #   951.447 μs (65574 allocations: 5.12 MiB)
    # render(scene_2_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 16) # 16 samples

    # # Using @paulmelis' style of hit(): @inbounds for i in eachindex(hittables) and Union{HitRecord, Nothing}
    # #   101.161 μs (4314 allocations: 459.41 KiB)
    # render(scene_2_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 1) # 1 sample

    # #render(scene_4_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 16)

    # # unchanged angle
    # @assert refract((@SVector[0.6,-0.8,0]), (@SVector[0.0,1.0,0.0]), 1.0) == @SVector[0.6,-0.8,0.0] 

    # # wider angle
    # t_refract_widerθ = refract(@SVector[0.6,-0.8,0.0], @SVector[0.0,1.0,0.0], 2.0)
    # @assert isapprox(t_refract_widerθ, @SVector[0.87519,-0.483779,0.0]; atol=1e-3)

    # # narrower angle
    # t_refract_narrowerθ = refract(@SVector[0.6,-0.8,0.0], @SVector[0.0,1.0,0.0], 0.5)
    # @assert isapprox(t_refract_narrowerθ, @SVector[0.3,-0.953939,0.0]; atol=1e-3)

    # #render(scene_diel_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 16)
    # #render(scene_diel_spheres(), default_camera(), 320, 32)

    # # Hollow Glass sphere using a negative radius
    # #render(scene_diel_spheres(-0.5; elem_type=ELEM_TYPE), t_default_cam, 96, 16)

    # #render(scene_diel_spheres(; elem_type=ELEM_TYPE), default_camera((SA{ELEM_TYPE}[-2,2,1]), (SA{ELEM_TYPE}[0,0,-1]),
    # #																 (SA{ELEM_TYPE}[0,1,0]), ELEM_TYPE(20)), 96, 16)

    # #render(scene_blue_red_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 16)


    # t_cam1 = default_camera([13,2,3], [0,0,0], [0,1,0], 20, 16/9, 0.1, 10.0; elem_type=ELEM_TYPE)

    # # took ~20s (really rough timing) in REPL, before optimization
    # # after optimization: 
    # #   880.997 ms (48801164 allocations: 847.10 MiB)
    # # after switching to Float32 + reducing allocations using rand_vec3!(): 
    # #    79.574 ms (  542467 allocations:  14.93 MiB)
    # # after optimizing skycolor, rand*, probably more stuff I forgot...
    # #    38.242 ms (  235752 allocations:   6.37 MiB)
    # # after removing all remaining Color, Vec3, replacing them with @SVector[]...
    # #    26.790 ms (   91486 allocations: 2.28 MiB)
    # # Using convert(Float32, ...) instead of MyFloat(...):
    # #    25.856 ms (   70650 allocations: 1.96 MiB)
    # # Don't specify return value of Option{HitRecord} in hit()
    # # Don't initialize unnecessary elements in HitRecord(): 
    # #    38.961 ms (   70681 allocations: 1.96 MiB) # WORSE, probably because we're writing a lot more to stack...?
    # # Replace MyFloat by Float32:
    # #    36.726 ms (13889 allocations: 724.09 KiB)
    # # @inline lots of stuff:
    # #    14.690 ms (13652 allocations: 712.98 KiB)
    # # rand(Float32) to avoid Float64s:
    # #    14.659 ms (13670 allocations: 713.84 KiB)
    # # Re-measured:
    # #    14.069 ms (13677 allocations: 714.22 KiB)
    # # After parameterized Vec3{T}:
    # # Float32: 14.422 ms (26376 allocations: 896.52 KiB) # claforte: why are there 2X the allocations as previously?
    # # Float64: 12.772 ms (12868 allocations: 1.08 MiB) (10% speed-up! Thanks @woclass!)
    # # rand() using MersenneTwister _rng w/ Float64:
    # #    14.092 ms (12769 allocations: 1.07 MiB) (SLOWER?!)
    # # rand() using Xoroshiro128Plus _rng w/ Float64:
    # #    12.581 ms (12943 allocations: 1.08 MiB)
    # # Above was all using 1 single thread. With 16 threads:
    # #     1.789 ms (12926 allocations: 1.08 MiB) (WOW!)
    # # Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
    # # Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
    # #     2.168 ms (13791 allocations: 1.15 MiB)
    # # Using  bunch of @inbounds, @simd in low-level functions
    # #     2.076 ms (13861 allocations: 1.15 MiB)
    # # Lots of optimizations, up to `Using non-mutable HitRecord, Union{HitRecordMissing}, ismissing():`
    # #     2.042 ms (14825 allocations: 1.23 MiB)
    # #@btime render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 96, 1)

    # # took 5020s in Pluto.jl, before optimizations!
    # # after lots of optimizations, up to switching to Float32 + reducing allocations using rand_vec3!(): 
    # #   10.862032 seconds (70.09 M allocations: 1.858 GiB, 1.53% gc time)
    # # after optimizing skycolor, rand*, probably more stuff I forgot...
    # #    4.926 s (25,694,163 allocations: 660.06 MiB)
    # # after removing all remaining Color, Vec3, replacing them with @SVector[]...
    # #    3.541 s (9074843 allocations: 195.22 MiB)
    # # Using convert(Float32, ...) instead of MyFloat(...):
    # #    3.222 s (2055106 allocations: 88.53 MiB)
    # # Don't specify return value of Option{HitRecord} in hit()
    # # Don't initialize unnecessary elements in HitRecord(): 
    # #    5.016 s (2056817 allocations: 88.61 MiB) #  WORSE, probably because we're writing a lot more to stack...?
    # # Replace MyFloat by Float32:
    # #    5.185 s (1819234 allocations: 83.55 MiB) # Increase of 1% is probably noise
    # # Remove normalize() in reflect() (that function assumes the inputs are normalized)
    # #    5.040 s (1832052 allocations: 84.13 MiB)
    # # @inline lots of stuff:
    # #    2.110 s (1823044 allocations: 83.72 MiB)
    # # rand(Float32) to avoid Float64s:
    # #    2.063 s (1777985 allocations: 81.66 MiB)
    # # @woclass's "Use alias instead of new struct", i.e. `const HittableList = Vector{Hittable}`:
    # #    1.934 s (1796954 allocations: 82.53 MiB)
    # # @woclass's Vec3{T} with T=Float64: (7.8% speed-up!)
    # #    1.800 s (1711061 allocations: 131.03 MiB) 
    # # rand() using Xoroshiro128Plus _rng w/ Float64:
    # #    1.808 s (1690104 allocations: 129.43 MiB) (i.e. rand is not a bottleneck?)
    # # Above was all using 1 single thread. With 16 threads:
    # #  265.331 ms (1645402 allocations: 126.02 MiB) (WOW!)
    # # Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
    # # Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
    # #  308.217 ms (1830162 allocations: 140.12 MiB)
    # # Using @inbounds, @simd in low-level functions:
    # #  302.952 ms (1892513 allocations: 144.88 MiB)
    # # Convert Camera and every Material structs to non-mutable:
    # #  301.042 ms (1849711 allocations: 141.61 MiB)  (i.e. - unchanged)
    # # Adapt @Christ_Foster's Base.getproperty w/ @inline @inbounds:
    # #  292.603 ms (1856398 allocations: 142.12 MiB) (ran multiple times, seems like real, 3-5% speed-up)
    # # Eliminate the off-by-half-a-pixel offset:
    # #  286.873 ms (1811412 allocations: 138.69 MiB) (ran multiple times, seems like ~2.5% speed-up)
    # # Fixed, per-thread RNGs with fixed seeds
    # #  286.575 ms (1884433 allocations: 144.26 MiB) (i.e. maybe a tiny bit faster considering this fixed seed has more allocations?)
    # # Make HitRecord non-mutable:
    # #   29.733 s (937962909 allocations: 69.88 GiB) (WTF!)
    # # Lots of optimizations, up to `Using non-mutable HitRecord, Union{HitRecordMissing}, ismissing():`
    # #  306.011 ms (1884433 allocations: 144.26 MiB) (Still slower... Hum)
    # # Using @paulmelis' style of hit(): @inbounds for i in eachindex(hittables) and Union{HitRecord, Nothing}
    # #  304.877 ms (1884433 allocations: 144.26 MiB)
    # # Extract the scene creation from the render() call:
    # #  300.344 ms (1883484 allocations: 144.21 MiB)
    # print("render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 200, 32):")
    # reseed!()
    # _scene_random_spheres = scene_random_spheres(; elem_type=ELEM_TYPE)
    # @btime render($_scene_random_spheres, $t_cam1, 200, 32) 

    # # After some optimization, took ~5.6 hours:
    # #   20171.646846 seconds (94.73 G allocations: 2.496 TiB, 1.06% gc time)
    # # ... however the image looked weird... too blurry
    # # After removing all remaining Color, Vec3, replacing them with @SVector[]... took ~3.7 hours:
    # #   13326.770907 seconds (29.82 G allocations: 714.941 GiB, 0.36% gc time)
    # # Using convert(Float32, ...) instead of MyFloat(...):
    # # Don't specify return value of Option{HitRecord} in hit()
    # # Don't initialize unnecessary elements in HitRecord(): 
    # # Took ~4.1 hours:
    # #   14723.339976 seconds (5.45 G allocations: 243.044 GiB, 0.11% gc time) # WORSE, probably because we're writing a lot more to stack...?
    # # Replace MyFloat by Float32:
    # # Lots of other optimizations including @inline lots of stuff: 
    # #    6018.101653 seconds (5.39 G allocations: 241.139 GiB, 0.21% gc time) (1.67 hour)
    # # @woclass's rand(Float32) to avoid Float64s: (expected to provide 2.2% speed up)
    # # @woclass's "Use alias instead of new struct", i.e. `const HittableList = Vector{Hittable}`
    # # @woclass's Vec3{T} with T=Float64: (7.8% speed-up!): 
    # #    5268.175362 seconds (4.79 G allocations: 357.005 GiB, 0.47% gc time) (1.46 hours)
    # # Above was all using 1 single thread. With 16 threads: (~20 minutes)
    # #    1210.363539 seconds (4.94 G allocations: 368.435 GiB, 10.08% gc time)
    # # Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
    # # Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
    # #    1298.522674 seconds (5.43 G allocations: 404.519 GiB, 10.18% gc time)
    # # Using @inbounds, @simd in low-level functions:
    # #    1314.510565 seconds (5.53 G allocations: 411.753 GiB, 10.21% gc time) # NOTE: difference due to randomness?
    # # Adapt @Christ_Foster's Base.getproperty w/ @inline @inbounds: (expect 3-5% speed-up)
    # # Eliminate the off-by-half-a-pixel offset: (expect ~2.5% speed-up)
    # # Fixed, per-thread RNGs with fixed seeds (expecting no noticeable change in speed)
    # #  Using 16 threads: (21m22s)
    # #    1282.437499 seconds (5.53 G allocations: 411.742 GiB, 10.08% gc time) (i.e. 2.5% speed-up... currently GC- and memory-bound?)
    # #  Using 14 threads: (21m45s)
    # #    1305.767627 seconds (5.53 G allocations: 411.741 GiB, 9.97% gc time)
    # #print("@time render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 1920, 1000):")
    # #@time render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 1920, 1000)


    # t_cam2 = default_camera([3,3,2], [0,0,-1], [0,1,0], 20, 16/9, 2.0, norm([3,3,2]-[0,0,-1]); 
    #                         elem_type=ELEM_TYPE)

    # # Before optimization:
    # #  5.993 s  (193097930 allocations: 11.92 GiB)
    # # after disabling: `Base.getproperty(vec::SVector{3}, sym::Symbol)`
    # #  1.001 s  ( 17406437 allocations: 425.87 MiB)
    # # after forcing Ray and point() to use Float64 instead of AbstractFloat:
    # #  397.905 ms (6269207 allocations: 201.30 MiB)
    # # after forcing use of Float32 instead of Float64:
    # #  487.680 ms (7128113 allocations: 196.89 MiB) # More allocations... something is causing them...
    # # after optimizing rand_vec3!/rand_vec2! to minimize allocations:
    # #  423.468 ms (6075725 allocations: 158.92 MiB)
    # # after optimizing skycolor, rand*, probably more stuff I forgot...
    # #  217.853 ms (2942272 allocations: 74.82 MiB)
    # # after removing all remaining Color, Vec3, replacing them with @SVector[]...
    # #   56.778 ms (1009344 allocations: 20.56 MiB)
    # # Using convert(Float32, ...) instead of MyFloat(...):
    # #   23.870 ms (210890 allocations: 8.37 MiB)
    # # Replace MyFloat by Float32:
    # #   22.390 ms (153918 allocations: 7.11 MiB)
    # # Various other changes, e.g. remove unnecessary normalize
    # #   20.241 ms (153792 allocations: 7.10 MiB)
    # # @inline lots of stuff:
    # #   18.065 ms (153849 allocations: 7.10 MiB)
    # # rand(Float32) to avoid Float64s:
    # #   16.035 ms (153777 allocations: 7.10 MiB)
    # # After @woclass's Vec3{T} with T=Float64:
    # #   16.822 ms (153591 allocations: 11.84 MiB)
    # # rand() using Xoroshiro128Plus _rng w/ Float64:
    # #   13.469 ms (153487 allocations: 11.83 MiB)
    # # Above was all using 1 single thread. With 16 threads:
    # #    6.537 ms (153599 allocations: 11.84 MiB)
    # # Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
    # # Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
    # #    6.766 ms (161000 allocations: 12.40 MiB)
    # # @inbounds and @simd in low-level functions
    # #    6.519 ms (160609 allocations: 12.37 MiB)
    # #render(scene_diel_spheres(; elem_type=ELEM_TYPE), t_cam2, 96, 16)

    # using Profile
    # render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 16, 1)
    # Profile.clear_malloc_data()
    # render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 17, 13)

end # @testset
