############################
# STEP 1 build executable binary
############################
FROM golang:alpine AS builder
ARG SERVICE_PATH='domains/problem'
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
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o /go/bin/${SERVICE_PATH} /go/src/server/${SERVICE_PATH}/cmd
RUN ls /go/bin

############################
# STEP 2 build a small image
############################
FROM scratch
# Copy our static executable.
COPY --from=builder /go/bin/$SERVICE_PATH /go/bin/$SERVICE_PATH
ENV IDLP_SERVICE_PATH=${SERVICE_PATH}

ENTRYPOINT ["/go/bin/domains/problem"]
