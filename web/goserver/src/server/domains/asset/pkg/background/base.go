package background

import (
	"context"
	"time"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
)

type AssetBackground interface {
	UpdateFromDisk(ctx context.Context, root string, timeout time.Duration)
}

type basicAssetBackground struct {
	Conn *rabbitmq.Connection
}

func NewBasicAssetBackground(conn *rabbitmq.Connection) AssetBackground {
	return &basicAssetBackground{
		conn,
	}
}

func New(conn *rabbitmq.Connection, middleware []Middleware) AssetBackground {
	var svc = NewBasicAssetBackground(conn)
	for _, m := range middleware {
		svc = m(svc)
	}
	return svc
}
