# Build
FROM rust:1.80-slim as build
WORKDIR /src
COPY Cargo.toml /src/Cargo.toml
RUN mkdir -p /src/src && echo 'fn main(){}' > /src/src/main.rs && cargo build --release
COPY src /src/src
RUN cargo build --release

# Runtime
FROM debian:stable-slim
WORKDIR /app
COPY --from=build /src/target/release/agents-wrangler-core /app/agents-wrangler-core
EXPOSE 8080
ENV RUST_LOG=info
CMD ["/app/agents-wrangler-core"]
