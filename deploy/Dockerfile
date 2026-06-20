# Ultra-lightweight Multi-Stage Dockerfile for Baremetal AI Engine
# Output image: <15 MB total footprint!

FROM alpine:3.19 AS builder
RUN apk add --no-cache gcc g++ make musl-dev openmp-dev
WORKDIR /build
COPY . .
RUN make all CFLAGS="-O3 -Wall -Wextra -std=c99 -fopenmp -Iinclude" LDFLAGS="-lm -fopenmp"

FROM alpine:3.19 AS runtime
RUN apk add --no-cache libgomp musl
WORKDIR /app
COPY --from=builder /build/inference /app/inference
COPY --from=builder /build/scripts/setup_model.py /app/scripts/setup_model.py
ENV OMP_NUM_THREADS=8
ENTRYPOINT ["/app/inference"]
CMD ["--help"]
