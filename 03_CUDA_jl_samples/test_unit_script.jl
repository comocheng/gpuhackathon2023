using CUDA
using BenchmarkTools

function rad_reduction_gpu!(radprod, radloss, radrts, corerxninds)
    radrts_gpu = cu(radrts)
    corerxninds_gpu = cu(corerxninds)

    r1 = radrts_gpu[corerxninds_gpu]

    t1 = (r1 .> 0).*1
    t2 = 1 .- t1

    prod = reduce(+, r1 .* t1)
    loss = reduce(+, r1 .* t2)

    print(prod)
    print(loss)
    
    radprod += prod
    radloss += loss

    return nothing
end

N = 2^20
radrts = reshape(repeat([1, -1], outer=div(N,2)), N)
corerxninds = collect(N)

radprod_gpu = 0.0
radloss_gpu = 0.0

@btime rad_reduction_gpu!(radprod_gpu, radloss_gpu, radrts, corerxninds)
print(radprod_gpu)
print(radloss_gpu)
