############################
# STEP 1 build executable binary
############################
FROM golang:alpine AS builder

# Install git + SSL ca certificates.
# Git is required for fetching the dependencies.
# Ca-certificates is required to call HTTPS endpoints.
RUN apk update && apk add --no-cache git ca-certificates && update-ca-certificates

WORKDIR /go/src/server

# Fetch dependencies.
COPY go.mod .
COPY go.sum .
RUN go mod download

COPY . .
# Build the binary.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o /go/bin/domains/cvat_task /go/src/server/domains/cvat_task/cmd

############################
# STEP 2 build a small image
############################
FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
# Copy our static executable.
COPY --from=builder /go/bin/domains/cvat_task /go/bin/domains/cvat_task

ENTRYPOINT ["/go/bin/domains/cvat_task"]
