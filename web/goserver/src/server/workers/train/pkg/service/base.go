package service

import (
	"context"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
)

type TrainModelService interface {
	RunCommands(ctx context.Context, req RunCommandsRequestData) (interface{}, error)
	GetGpuAmount(ctx context.Context, req GetGpuAmountRequestData) (interface{}, error)
}

type basicTrainModelService struct {
	Conn *rabbitmq.Connection
}

// NewBasicApiService returns a naive, stateless implementation of ApiService.
func NewBasicAssetService(conn *rabbitmq.Connection) TrainModelService {
	return &basicTrainModelService{
		conn,
	}
}

// New returns a ApiService with all of the expected middleware wired in.
func New(conn *rabbitmq.Connection, middleware []Middleware) TrainModelService {
	var svc = NewBasicAssetService(conn)
	for _, m := range middleware {
		svc = m(svc)
	}
	return svc
}
