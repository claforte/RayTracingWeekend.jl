module RayTracingWeekend

using Images
using LinearAlgebra
using Random
using RandomNumbers.Xorshifts
using StaticArrays

include("vec.jl")
export Vec3, squared_length, near_zero, rgb, rgb_gamma2
include("init.jl")
export TRNG
include("structs.jl")
export Ray, ray, Hittable, HittableList, Material, HitRecord, Scatter
include("rand.jl")
export reseed!, trand, random_vec3_in_sphere, random_between, random_vec3, random_vec2, 
	random_vec3_on_sphere, random_vec2_in_disk
include("shapes.jl")
export Sphere, MovingSphere, sphere
include("aabb.jl")
export Aabb
include("hit.jl")
export point, ray_to_HitRecord, hit
include("ray_color.jl")
export skycolor, color_vec3_in_rgb, ray_color
include("camera.jl")
export Camera, default_camera, get_ray
include("render.jl")
export render
include("light.jl") # light transport
export reflect, refract, reflectance
include("material.jl")
export Lambertian, scatter, Metal, Dielectric
include("scenes.jl")
export scene_2_spheres, scene_4_spheres, scene_blue_red_spheres, scene_diel_spheres, scene_random_spheres

end
