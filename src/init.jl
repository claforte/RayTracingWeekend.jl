# Per-thread Random Number Generator. Initialized later...
const TRNG = Xoroshiro128Plus[]

function __init__()
	# Instantiate 1 RNG (Random Number Generator) per thread, for performance.
	# This can't be done during precompilation since the number of threads isn't known then.
	resize!(TRNG, Threads.nthreads())
	for i in 1:Threads.nthreads()
		TRNG[i] = Xoroshiro128Plus(i)
	end
	nothing
end
