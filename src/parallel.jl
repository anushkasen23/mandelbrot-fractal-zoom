# parallel.jl
# Parallel Mandelbrot fractal renderer using Distributed & RemoteChannel 
# Author: Anushka Sen
# Notes: dynamic work queue (one frame per job). Keeps sequential mandelbrot logic intact.

using Distributed
@everywhere include("sequential.jl") # Including sequential everywhere as all workers use the run_seq() function

#=============================================================================
                            YOUR IMPLEMENTATION HERE
=============================================================================#

# Master 
function generate_fractal_par!(FP::FractalParams, data::Array{Float64, 3})

    N, F, max_iters, center, alpha, delta = extract_params(FP)

    P = nworkers() # P + 1 processors - 1 Master + P Workers

#     if P == 0
#     @warn "No workers detected (nworkers()==0). fall back to sequential run."
#     return generate_fractal_seq!(FP, data)
# end

    # t will be the total time spent in this code
    t = @elapsed begin  

        # Master logic here

        job_channel = RemoteChannel(() -> Channel{Int}(F))
        result_channel = RemoteChannel(() -> Channel{Tuple{Int, Array{Float64,2}}}(F))

        for f in 1:F
            put!(job_channel, f)
        end
        close(job_channel)

        worker_futures = []
        for pid in workers()
            future = @spawnat pid work(FP, job_channel, result_channel)
            push!(worker_futures, future)
        end

        for f in 1:F
            frame_id, frame_data = take!(result_channel)
           # println("master receive frame $frame_id ")
            data[:, :, frame_id] = frame_data
        end

        for future in worker_futures
            wait(future)
        end
        
    end
    
    return t 

end

# Worker
@everywhere function work(FP::FractalParams, job_channel::RemoteChannel, result_channel::RemoteChannel)

    N, F, max_iters, center, alpha, delta = extract_params(FP)

    # Worker logic here

    while true
        frame_id = try
            take!(job_channel)
        catch
            break
        end
            
        frame = zeros(Float64, N, N)

        local_delta = delta * alpha^(frame_id-1) 
        x_min = center[1] - local_delta 
        y_min = center[2] - local_delta
        dw = (2 * local_delta) / N

        @inbounds for j in 1:N              # Columns
            y = y_min + (j - 1) * dw        
            @inbounds for i in 1:N          # Rows
                x = x_min + (i - 1) * dw
                c = Complex(x, y)
                frame[i, j] = mandelbrot(c, max_iters) 
            end
        end
        
        # println("worker $(myid()) finished frame $frame_id")

        put!(result_channel, (frame_id, frame))
    
    end

    println("Worker $(myid()) slept")
    return nothing
end

#=============================================================================
                            YOUR IMPLEMENTATION HERE
=============================================================================#

function run_par(FP::FractalParams)
    data = zeros(Float64, FP.N, FP.N, FP.F)
    t = generate_fractal_par!(FP, data)
    return t, data
end


