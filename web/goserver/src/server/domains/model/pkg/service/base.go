package service

import (
	"context"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"

	kitendpoint "server/kit/endpoint"
)

type ModelService interface {
	CreateFromGeneric(ctx context.Context, req CreateFromGenericRequest) chan kitendpoint.Response
	Delete(ctx context.Context, req DeleteRequestData, responseChan chan kitendpoint.Response)
	Evaluate(ctx context.Context, req EvaluateRequest) chan kitendpoint.Response
	FineTune(ctx context.Context, req FineTuneRequestData) chan kitendpoint.Response
	List(ctx context.Context, req ListRequestData) chan kitendpoint.Response
	UpdateFromLocal(ctx context.Context, req UpdateFromLocalRequestData) chan kitendpoint.Response
}

type basicModelService struct {
	Conn          *rabbitmq.Connection
	problemPath   string
	trainingsPath string
}

func NewBasicModelService(conn *rabbitmq.Connection, problemPath, trainingsPath string) ModelService {
	return &basicModelService{
		Conn:          conn,
		problemPath:   problemPath,
		trainingsPath: trainingsPath,
	}
}

func New(conn *rabbitmq.Connection, problemPath, trainingsPath string, middleware []Middleware) ModelService {
	var svc = NewBasicModelService(conn, problemPath, trainingsPath)
	for _, m := range middleware {
		svc = m(svc)
	}
	return svc
}
