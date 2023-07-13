function sequential_add!(y, x)
    
    for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing

end

function parallel_add!(y, x)

    Threads.@threads for i in eachindex(y, x)
        @inbounds y[i] += x[i]
    end
    return nothing
end

N = 2^24
x = fill(1.0f0, N)
y = fill(2.0f0, N)


using CUDA
xd = CUDA.fill(1.0f0, N)

function add_broadcast!(y, x)
    CUDA.@sync y .+= x
    return
end

function gpu_add1!(y, x)
    for i = 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function gpu_add2!(y, x)
    index = threadIdx().x
    stride = blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return
end

function bench_gpu1!(y, x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
end

function bench_gpu2!(y, x)
    CUDA.@sync begin
        @cuda threads=256 gpu_add2!(y, x)
    end
end

function bench_gpu3!(y, x)
    numblocks = ceil(Int, length(y)/256)
    CUDA.@sync begin
        @cuda threads=256 blocks=numblocks gpu_add3!(y, x)
    end 
end

function bench_gpu4!(y, x)
    kernel = @cuda launch=false gpu_add3!(y, x)
    config = launch_configuration(kernel.fun)
    threads = min(length(y), config.threads)
    blocks = cld(length(y), threads)

    CUDA.@sync begin
        kernel(y, x; threads, blocks)
    end
end

using BenchmarkTools

fill!(y, 2)
@btime sequential_add!($y, $x)
fill!(y, 2)
@btime parallel_add!($y, $x)
yd = CUDA.fill(2.0f0, N)
@btime add_broadcast!($yd, $xd)
fill!(yd, 2)
@btime bench_gpu1!($yd, $xd)
fill!(yd, 2)
@btime bench_gpu2!($yd, $xd)
fill!(yd, 2)
@btime bench_gpu3!($yd, $xd)
fill!(yd, 2)
@btime bench_gpu4!($yd, $xd)
