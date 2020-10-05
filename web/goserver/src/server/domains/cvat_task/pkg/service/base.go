package service

import (
	"context"
	kitendpoint "server/kit/endpoint"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
)

type CvatTaskService interface {
	FindInFolder(ctx context.Context, req FindInFolderRequestData) (result FindInFolderResponseData)
	Setup(ctx context.Context, req SetupRequestData) chan kitendpoint.Response
	Dump(ctx context.Context, req DumpRequestData) chan kitendpoint.Response
}

type basicCvatTaskService struct {
	Conn *rabbitmq.Connection
}

func NewBasicBuildService(conn *rabbitmq.Connection) CvatTaskService {
	return &basicCvatTaskService{
		Conn: conn,
	}
}

func New(conn *rabbitmq.Connection, middleware []Middleware) CvatTaskService {
	var svc = NewBasicBuildService(conn)
	for _, m := range middleware {
		svc = m(svc)
	}
	return svc
}
