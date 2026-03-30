# syntax=docker/dockerfile:1
FROM golang:1.21-alpine AS builder

# Set destination for COPY
WORKDIR /app

# Download Go modules
COPY go.mod go.sum ./
RUN go mod download

# Copy the source code. Note the slash at the end, as explained in
# https://docs.docker.com/engine/reference/builder/#copy
COPY *.go ./
COPY rag/ ./rag/

# Build
RUN CGO_ENABLED=0 GOOS=linux go build -o /judgement-rag-go

# Final stage
FROM alpine:latest

# Install dependencies (tesseract could be added here if OS exec was used)
RUN apk add --no-cache ca-certificates tesseract-ocr poppler-utils

WORKDIR /app
COPY --from=builder /judgement-rag-go /app/judgement-rag-go

# Expose port
EXPOSE 8000

# Run
CMD ["/app/judgement-rag-go"]
