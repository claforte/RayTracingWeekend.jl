"""
To reproduce crash:
1. Start Julia with some threads, e.g. 
   `julia --project=. --threads=8 repro_crash.jl`
2. Uncomment the @threads in RayTracingWeekend.jl:

```
#Threads.@threads # claforte: uncomment for CRASH?!
	for i in 1:image_height
```

by 

```
Threads.@threads for i in 1:image_height
```

Typical segmentation fault:

claforte@ub-claforte:~/.julia/depot/dev/RayTracingWeekend/src/proto$ julia --project=. --threads=8 repro_crash.jl 

signal (11): Segmentation fault
in expression starting at /home/claforte/.julia/depot/dev/RayTracingWeekend/src/proto/repro_crash.jl:6
jl_lookup_generic_ at /buildworker/worker/package_linux64/build/src/gf.c:2347 [inlined]
jl_apply_generic at /buildworker/worker/package_linux64/build/src/gf.c:2415
wait at ./task.jl:322 [inlined]
threading_run at ./threadingconstructs.jl:34
macro expansion at ./threadingconstructs.jl:93 [inlined]
render at /home/claforte/.julia/depot/dev/RayTracingWeekend/src/RayTracingWeekend.jl:331
unknown function (ip: 0x7ff85b9985f7)
_jl_invoke at /buildworker/worker/package_linux64/build/src/gf.c:2237 [inlined]
jl_apply_generic at /buildworker/worker/package_linux64/build/src/gf.c:2419
jl_apply at /buildworker/worker/package_linux64/build/src/julia.h:1703 [inlined]
do_call at /buildworker/worker/package_linux64/build/src/interpreter.c:115
eval_value at /buildworker/worker/package_linux64/build/src/interpreter.c:204
eval_stmt_value at /buildworker/worker/package_linux64/build/src/interpreter.c:155 [inlined]
eval_body at /buildworker/worker/package_linux64/build/src/interpreter.c:562
jl_interpret_toplevel_thunk at /buildworker/worker/package_linux64/build/src/interpreter.c:670
jl_toplevel_eval_flex at /buildworker/worker/package_linux64/build/src/toplevel.c:877
jl_toplevel_eval_flex at /buildworker/worker/package_linux64/build/src/toplevel.c:825
jl_toplevel_eval_in at /buildworker/worker/package_linux64/build/src/toplevel.c:929
eval at ./boot.jl:360 [inlined]
include_string at ./loading.jl:1094
_jl_invoke at /buildworker/worker/package_linux64/build/src/gf.c:2237 [inlined]
jl_apply_generic at /buildworker/worker/package_linux64/build/src/gf.c:2419
_include at ./loading.jl:1148
include at ./Base.jl:386
_jl_invoke at /buildworker/worker/package_linux64/build/src/gf.c:2237 [inlined]
jl_apply_generic at /buildworker/worker/package_linux64/build/src/gf.c:2419
exec_options at ./client.jl:285
_start at ./client.jl:485
jfptr__start_34289.clone_1 at /home/claforte/.julia/lib/julia/sys.so (unknown line)
_jl_invoke at /buildworker/worker/package_linux64/build/src/gf.c:2237 [inlined]
jl_apply_generic at /buildworker/worker/package_linux64/build/src/gf.c:2419
jl_apply at /buildworker/worker/package_linux64/build/src/julia.h:1703 [inlined]
true_main at /buildworker/worker/package_linux64/build/src/jlapi.c:560
repl_entrypoint at /buildworker/worker/package_linux64/build/src/jlapi.c:702
main at julia (unknown line)
__libc_start_main at /lib/x86_64-linux-gnu/libc.so.6 (unknown line)
unknown function (ip: 0x4007d8)
Allocations: 12282419 (Pool: 12278054; Big: 4365); GC: 5
Segmentation fault (core dumped)

"""


using RayTracingWeekend
using StaticArrays
ELEM_TYPE = Float64
t_default_cam = default_camera(SA{ELEM_TYPE}[0,0,0])
render(scene_2_spheres(; elem_type=ELEM_TYPE), t_default_cam, 96, 16)
