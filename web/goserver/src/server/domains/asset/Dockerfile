############################
# STEP 1 build executable binary
############################
FROM golang:alpine AS builder
# Install git.
# Git is required for fetching the dependencies.
RUN apk update && apk add --no-cache git
WORKDIR /go/src/server

# Fetch dependencies.
COPY go.mod .
COPY go.sum .
RUN go mod download

COPY . .
# Build the binary.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o /go/bin/domains/asset /go/src/server/domains/asset/cmd

############################
# STEP 2 build a small image
############################
FROM scratch
# Copy our static executable.
COPY --from=builder /go/bin/domains/asset /go/bin/domains/asset

ENTRYPOINT ["/go/bin/domains/asset"]
