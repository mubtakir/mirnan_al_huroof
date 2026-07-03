using FFTW

PHASE_DIM = 9958
v = rand(PHASE_DIM)

println("Timing normal fft...")
# Warmup
fft(v)
t1 = time_ns()
for i in 1:1000
    fft(v)
end
t2 = time_ns()
println("Normal fft: $((t2 - t1)/1e6 / 1000) ms")

println("Planning fft...")
plan = plan_fft(zeros(ComplexF64, PHASE_DIM))
v_complex = ComplexF64.(v)

# Warmup
plan * v_complex
t3 = time_ns()
for i in 1:1000
    plan * v_complex
end
t4 = time_ns()
println("Planned fft: $((t4 - t3)/1e6 / 1000) ms")
