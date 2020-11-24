package service

import (
	"context"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"

	t "server/db/pkg/types"
	kitendpoint "server/kit/endpoint"
)

type BuildService interface {
	Create(ctx context.Context, req CreateRequestData)
	CreateEmpty(ctx context.Context, req CreateEmptyRequestData) t.Build
	List(ctx context.Context, req ListRequestData) chan kitendpoint.Response
	UpdateAssetState(ctx context.Context, req UpdateAssetStateRequestData) UpdateAssetStateResponseData
	UpdateTmps(ctx context.Context, req UpdateTmpsRequestData) UpdateTmpsResponseData
}

type basicBuildService struct {
	Conn        *rabbitmq.Connection
	ProblemPath string
}

func NewBasicBuildService(conn *rabbitmq.Connection, problemPath string) BuildService {
	return &basicBuildService{
		Conn:        conn,
		ProblemPath: problemPath,
	}
}

func New(conn *rabbitmq.Connection, problemPath string, middleware []Middleware) BuildService {
	var svc = NewBasicBuildService(conn, problemPath)
	for _, m := range middleware {
		svc = m(svc)
	}
	return svc
}
