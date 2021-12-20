# Prototype - copied from pluto_RayTracingWeekend.jl
# Adapted from [Ray Tracing In One Weekend by Peter Shirley](https://raytracing.github.io/books/RayTracingInOneWeekend.html) and 
# [cshenton's Julia implementation](https://github.com/cshenton/RayTracing.jl)"

using RayTracingWeekend

using BenchmarkTools
using Images
using InteractiveUtils
using LinearAlgebra
using StaticArrays
using LoopVectorization: @turbo

# Float32/Float64
ELEM_TYPE = Float64

t_default_cam = default_camera(SA{ELEM_TYPE}[0,0,0])

t_cam1 = default_camera([13,2,3], [0,0,0], [0,1,0], 20, 16/9, 0.1, 10.0; elem_type=ELEM_TYPE)

t_cam2 = default_camera([3,3,2], [0,0,-1], [0,1,0], 20, 16/9, 2.0, norm([3,3,2]-[0,0,-1]); 
						elem_type=ELEM_TYPE)

# After some optimization:
#  46.506 ms (917106 allocations: 16.33 MiB)
# Using convert(Float32, ...) instead of MyFloat(...):
#  14.137 ms (118583 allocations: 4.14 MiB)
# Don't specify return value of Option{HitRecord} in hit()
#  24.842 ms (527738 allocations: 16.63 MiB)
# Don't initialize unnecessary elements in HitRecord(): 
#  14.862 ms (118745 allocations: 4.15 MiB)  (but computer was busy...)
# Replace MyFloat by Float32:
#  11.792 ms ( 61551 allocations: 2.88 MiB)
# Remove ::HitRecord return value in remaining hit() method:
#  11.545 ms ( 61654 allocations: 2.88 MiB)
# with mutable HitRecord
#  11.183 ms ( 61678 allocations: 2.88 MiB) (insignificant?)
# @inline tons of stuff. Note: render() uses `for i in 1:image_height, j in 1:image_width`,
#    i.e. iterating 1 row at time!
#   8.129 ms ( 61660 allocations: 2.88 MiB)
# Using in render(): `for j in 1:image_width, i in 1:image_height # iterate over each column`
#  10.489 ms ( 61722 allocations: 2.88 MiB) (consistently slower!)
# ... sticking with `for i in 1:image_height, j in 1:image_width # iterate over each row` for now...
# Using `get_ray(cam, u+δu, v+δv)` (fixes minor bug, extract constants outside inner loop):
# ... performance appears equivalent, maybe a tiny bit faster on avg (1%?)
# Re-measured:
#   8.077 ms (61610 allocations: 2.88 MiB)
# After parameterized Vec3{T}:
# Float64: 8.344 ms (61584 allocations: 4.82 MiB)
# Float32: 8.750 ms (123425 allocations: 3.83 MiB)
# rand() using MersenneTwister _rng w/ Float64:
#   6.967 ms (61600 allocations: 4.82 MiB)
# rand() using Xoroshiro128Plus _rng w/ Float64:
#   6.536 ms (61441 allocations: 4.81 MiB)
# Above was all using 1 single thread. With 16 threads:
#   4.414 ms (61673 allocations: 4.82 MiB)
# Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
# Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
#   4.465 ms (65680 allocations: 5.13 MiB)
# Lots of optimizations... ending with make HitRecord non-mutable:
#   2.225 ms (445188 allocations: 34.08 MiB)
# Using non-mutable HitRecord, Union{HitRecord,Missing}, ismissing():
#   976.365 μs (65574 allocations: 5.12 MiB)
# Using @paulmelis' style of hit(): @inbounds for i in eachindex(hittables) and Union{HitRecord, Nothing}
#   951.447 μs (65574 allocations: 5.12 MiB)
render(scene_2_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 16) # 16 samples

# Iterate over each column: 614.820 μs
# Iterate over each row: 500.334 μs
# With Rand(Float32) everywhere:
#   489.745 μs (3758 allocations: 237.02 KiB)
# With parameterized Vec3{T}:
# Float64: 530.957 μs (3760 allocations: 414.88 KiB)
# rand() using MersenneTwister _rng w/ Float64:
#   473.672 μs (3748 allocations: 413.94 KiB)
# rand() using Xoroshiro128Plus _rng w/ Float64:
#   444.399 μs (3737 allocations: 413.08 KiB)
# Above was all using 1 single thread. With 16 threads:
#   300.438 μs (3829 allocations: 420.86 KiB)
# Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
# Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
#   314.094 μs (4009 allocations: 434.97 KiB)
# Lots of optimizations... ending with make HitRecord non-mutable:
#   136.388 μs (28306 allocations: 2.28 MiB)
# Using non-mutable HitRecord, Union{HitRecordMissing}, ismissing():
#   102.764 μs (4314 allocations: 459.41 KiB)
# Using @paulmelis' style of hit(): @inbounds for i in eachindex(hittables) and Union{HitRecord, Nothing}
#   101.161 μs (4314 allocations: 459.41 KiB)
render(scene_2_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 1) # 1 sample

#render(scene_4_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 16)


#render(scene_diel_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 16)
#render(scene_diel_spheres(), default_camera(), 320, 32)

# Hollow Glass sphere using a negative radius
#render(scene_diel_spheres(-0.5; elem_type=ELEM_TYPE), t_default_cam, 96, 16)

#render(scene_diel_spheres(; elem_type=ELEM_TYPE), default_camera((SA{ELEM_TYPE}[-2,2,1]), (SA{ELEM_TYPE}[0,0,-1]),
#																 (SA{ELEM_TYPE}[0,1,0]), ELEM_TYPE(20)), 96, 16)


#render(scene_blue_red_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 16)

# took ~20s (really rough timing) in REPL, before optimization
# after optimization: 
#   880.997 ms (48801164 allocations: 847.10 MiB)
# after switching to Float32 + reducing allocations using rand_vec3!(): 
#    79.574 ms (  542467 allocations:  14.93 MiB)
# after optimizing skycolor, rand*, probably more stuff I forgot...
#    38.242 ms (  235752 allocations:   6.37 MiB)
# after removing all remaining Color, Vec3, replacing them with @SVector[]...
#    26.790 ms (   91486 allocations: 2.28 MiB)
# Using convert(Float32, ...) instead of MyFloat(...):
#    25.856 ms (   70650 allocations: 1.96 MiB)
# Don't specify return value of Option{HitRecord} in hit()
# Don't initialize unnecessary elements in HitRecord(): 
#    38.961 ms (   70681 allocations: 1.96 MiB) # WORSE, probably because we're writing a lot more to stack...?
# Replace MyFloat by Float32:
#    36.726 ms (13889 allocations: 724.09 KiB)
# @inline lots of stuff:
#    14.690 ms (13652 allocations: 712.98 KiB)
# rand(Float32) to avoid Float64s:
#    14.659 ms (13670 allocations: 713.84 KiB)
# Re-measured:
#    14.069 ms (13677 allocations: 714.22 KiB)
# After parameterized Vec3{T}:
# Float32: 14.422 ms (26376 allocations: 896.52 KiB) # claforte: why are there 2X the allocations as previously?
# Float64: 12.772 ms (12868 allocations: 1.08 MiB) (10% speed-up! Thanks @woclass!)
# rand() using MersenneTwister _rng w/ Float64:
#    14.092 ms (12769 allocations: 1.07 MiB) (SLOWER?!)
# rand() using Xoroshiro128Plus _rng w/ Float64:
#    12.581 ms (12943 allocations: 1.08 MiB)
# Above was all using 1 single thread. With 16 threads:
#     1.789 ms (12926 allocations: 1.08 MiB) (WOW!)
# Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
# Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
#     2.168 ms (13791 allocations: 1.15 MiB)
# Using  bunch of @inbounds, @simd in low-level functions
#     2.076 ms (13861 allocations: 1.15 MiB)
# Lots of optimizations, up to `Using non-mutable HitRecord, Union{HitRecordMissing}, ismissing():`
#     2.042 ms (14825 allocations: 1.23 MiB)
#@btime render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 96, 1)

# took 5020s in Pluto.jl, before optimizations!
# after lots of optimizations, up to switching to Float32 + reducing allocations using rand_vec3!(): 
#   10.862032 seconds (70.09 M allocations: 1.858 GiB, 1.53% gc time)
# after optimizing skycolor, rand*, probably more stuff I forgot...
#    4.926 s (25,694,163 allocations: 660.06 MiB)
# after removing all remaining Color, Vec3, replacing them with @SVector[]...
#    3.541 s (9074843 allocations: 195.22 MiB)
# Using convert(Float32, ...) instead of MyFloat(...):
#    3.222 s (2055106 allocations: 88.53 MiB)
# Don't specify return value of Option{HitRecord} in hit()
# Don't initialize unnecessary elements in HitRecord(): 
#    5.016 s (2056817 allocations: 88.61 MiB) #  WORSE, probably because we're writing a lot more to stack...?
# Replace MyFloat by Float32:
#    5.185 s (1819234 allocations: 83.55 MiB) # Increase of 1% is probably noise
# Remove normalize() in reflect() (that function assumes the inputs are normalized)
#    5.040 s (1832052 allocations: 84.13 MiB)
# @inline lots of stuff:
#    2.110 s (1823044 allocations: 83.72 MiB)
# rand(Float32) to avoid Float64s:
#    2.063 s (1777985 allocations: 81.66 MiB)
# @woclass's "Use alias instead of new struct", i.e. `const HittableList = Vector{Hittable}`:
#    1.934 s (1796954 allocations: 82.53 MiB)
# @woclass's Vec3{T} with T=Float64: (7.8% speed-up!)
#    1.800 s (1711061 allocations: 131.03 MiB) 
# rand() using Xoroshiro128Plus _rng w/ Float64:
#    1.808 s (1690104 allocations: 129.43 MiB) (i.e. rand is not a bottleneck?)
# Above was all using 1 single thread. With 16 threads:
#  265.331 ms (1645402 allocations: 126.02 MiB) (WOW!)
# Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
# Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
#  308.217 ms (1830162 allocations: 140.12 MiB)
# Using @inbounds, @simd in low-level functions:
#  302.952 ms (1892513 allocations: 144.88 MiB)
# Convert Camera and every Material structs to non-mutable:
#  301.042 ms (1849711 allocations: 141.61 MiB)  (i.e. - unchanged)
# Adapt @Christ_Foster's Base.getproperty w/ @inline @inbounds:
#  292.603 ms (1856398 allocations: 142.12 MiB) (ran multiple times, seems like real, 3-5% speed-up)
# Eliminate the off-by-half-a-pixel offset:
#  286.873 ms (1811412 allocations: 138.69 MiB) (ran multiple times, seems like ~2.5% speed-up)
# Fixed, per-thread RNGs with fixed seeds
#  286.575 ms (1884433 allocations: 144.26 MiB) (i.e. maybe a tiny bit faster considering this fixed seed has more allocations?)
# Make HitRecord non-mutable:
#   29.733 s (937962909 allocations: 69.88 GiB) (WTF!)
# Lots of optimizations, up to `Using non-mutable HitRecord, Union{HitRecordMissing}, ismissing():`
#  306.011 ms (1884433 allocations: 144.26 MiB) (Still slower... Hum)
# Using @paulmelis' style of hit(): @inbounds for i in eachindex(hittables) and Union{HitRecord, Nothing}
#  304.877 ms (1884433 allocations: 144.26 MiB)
# Extract the scene creation from the render() call:
#  300.344 ms (1883484 allocations: 144.21 MiB)
# Separate the module's reusable functions/classes from this proto.jl
#  296.824 ms (min) (1909996 allocs: 146.23 MiB)
print("render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 200, 32):")
reseed!()
_scene_random_spheres = scene_random_spheres(; elem_type=ELEM_TYPE)
@benchmark render($_scene_random_spheres, $t_cam1, 200, 32) 

# After some optimization, took ~5.6 hours:
#   20171.646846 seconds (94.73 G allocations: 2.496 TiB, 1.06% gc time)
# ... however the image looked weird... too blurry
# After removing all remaining Color, Vec3, replacing them with @SVector[]... took ~3.7 hours:
#   13326.770907 seconds (29.82 G allocations: 714.941 GiB, 0.36% gc time)
# Using convert(Float32, ...) instead of MyFloat(...):
# Don't specify return value of Option{HitRecord} in hit()
# Don't initialize unnecessary elements in HitRecord(): 
# Took ~4.1 hours:
#   14723.339976 seconds (5.45 G allocations: 243.044 GiB, 0.11% gc time) # WORSE, probably because we're writing a lot more to stack...?
# Replace MyFloat by Float32:
# Lots of other optimizations including @inline lots of stuff: 
#    6018.101653 seconds (5.39 G allocations: 241.139 GiB, 0.21% gc time) (1.67 hour)
# @woclass's rand(Float32) to avoid Float64s: (expected to provide 2.2% speed up)
# @woclass's "Use alias instead of new struct", i.e. `const HittableList = Vector{Hittable}`
# @woclass's Vec3{T} with T=Float64: (7.8% speed-up!): 
#    5268.175362 seconds (4.79 G allocations: 357.005 GiB, 0.47% gc time) (1.46 hours)
# Above was all using 1 single thread. With 16 threads: (~20 minutes)
#    1210.363539 seconds (4.94 G allocations: 368.435 GiB, 10.08% gc time)
# Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
# Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
#    1298.522674 seconds (5.43 G allocations: 404.519 GiB, 10.18% gc time)
# Using @inbounds, @simd in low-level functions:
#    1314.510565 seconds (5.53 G allocations: 411.753 GiB, 10.21% gc time) # NOTE: difference due to randomness?
# Adapt @Christ_Foster's Base.getproperty w/ @inline @inbounds: (expect 3-5% speed-up)
# Eliminate the off-by-half-a-pixel offset: (expect ~2.5% speed-up)
# Fixed, per-thread RNGs with fixed seeds (expecting no noticeable change in speed)
#  Using 16 threads: (21m22s)
#    1282.437499 seconds (5.53 G allocations: 411.742 GiB, 10.08% gc time) (i.e. 2.5% speed-up... currently GC- and memory-bound?)
#  Using 14 threads: (21m45s)
#    1305.767627 seconds (5.53 G allocations: 411.741 GiB, 9.97% gc time)
#print("@time render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 1920, 1000):")
#@time render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 1920, 1000)

# Before optimization:
#  5.993 s  (193097930 allocations: 11.92 GiB)
# after disabling: `Base.getproperty(vec::SVector{3}, sym::Symbol)`
#  1.001 s  ( 17406437 allocations: 425.87 MiB)
# after forcing Ray and point() to use Float64 instead of AbstractFloat:
#  397.905 ms (6269207 allocations: 201.30 MiB)
# after forcing use of Float32 instead of Float64:
#  487.680 ms (7128113 allocations: 196.89 MiB) # More allocations... something is causing them...
# after optimizing rand_vec3!/rand_vec2! to minimize allocations:
#  423.468 ms (6075725 allocations: 158.92 MiB)
# after optimizing skycolor, rand*, probably more stuff I forgot...
#  217.853 ms (2942272 allocations: 74.82 MiB)
# after removing all remaining Color, Vec3, replacing them with @SVector[]...
#   56.778 ms (1009344 allocations: 20.56 MiB)
# Using convert(Float32, ...) instead of MyFloat(...):
#   23.870 ms (210890 allocations: 8.37 MiB)
# Replace MyFloat by Float32:
#   22.390 ms (153918 allocations: 7.11 MiB)
# Various other changes, e.g. remove unnecessary normalize
#   20.241 ms (153792 allocations: 7.10 MiB)
# @inline lots of stuff:
#   18.065 ms (153849 allocations: 7.10 MiB)
# rand(Float32) to avoid Float64s:
#   16.035 ms (153777 allocations: 7.10 MiB)
# After @woclass's Vec3{T} with T=Float64:
#   16.822 ms (153591 allocations: 11.84 MiB)
# rand() using Xoroshiro128Plus _rng w/ Float64:
#   13.469 ms (153487 allocations: 11.83 MiB)
# Above was all using 1 single thread. With 16 threads:
#    6.537 ms (153599 allocations: 11.84 MiB)
# Above was all using max bounces=4, since this looked fine to me (except the negatively scaled sphere). 
# Switching to max bounces=16 to match C++ version decreased performance by 7.2%:
#    6.766 ms (161000 allocations: 12.40 MiB)
# @inbounds and @simd in low-level functions
#    6.519 ms (160609 allocations: 12.37 MiB)
#render(scene_diel_spheres(; elem_type=ELEM_TYPE), t_cam2, 96, 16)

using Profile
render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 16, 1)
Profile.clear_malloc_data()
render(scene_random_spheres(; elem_type=ELEM_TYPE), t_cam1, 320, 64)

