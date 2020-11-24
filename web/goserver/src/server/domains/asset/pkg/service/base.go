package service

import (
	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
)

type AssetService interface {
}

type basicAssetService struct {
	Conn *rabbitmq.Connection
}

func NewBasicAssetService(conn *rabbitmq.Connection) AssetService {
	return &basicAssetService{
		conn,
	}
}

func New(conn *rabbitmq.Connection, middleware []Middleware) AssetService {
	var svc = NewBasicAssetService(conn)
	for _, m := range middleware {
		svc = m(svc)
	}
	return svc
}
