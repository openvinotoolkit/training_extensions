package service

import (
	"context"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"

	kitendpoint "server/kit/endpoint"
)

type ProblemService interface {
	Create(ctx context.Context, req CreateRequestData, responseChan chan kitendpoint.Response)
	Details(ctx context.Context, req DetailsRequestData, responseChan chan kitendpoint.Response)
	List(ctx context.Context, req ListRequestData, responseChan chan kitendpoint.Response)
	UpdateFromLocal(ctx context.Context, req UpdateFromLocalRequestData, responseChan chan kitendpoint.Response)
}

type basicProblemService struct {
	Conn          *rabbitmq.Connection
	problemPath   string
	trainingsPath string
}

// NewBasicApiService returns a naive, stateless implementation of ApiService.
func NewBasicProblemService(conn *rabbitmq.Connection, problemPath, trainingsPath string) ProblemService {
	return &basicProblemService{
		conn,
		problemPath,
		trainingsPath,
	}
}

// New returns a ApiService with all of the expected middleware wired in.
func New(conn *rabbitmq.Connection, problemPath, trainingsPath string, middleware []Middleware) ProblemService {
	var svc = NewBasicProblemService(conn, problemPath, trainingsPath)
	for _, m := range middleware {
		svc = m(svc)
	}
	return svc
}
