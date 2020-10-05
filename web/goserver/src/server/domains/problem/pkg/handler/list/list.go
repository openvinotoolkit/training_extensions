package list

import (
	"context"
	"encoding/json"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	n "server/common/names"
	problemFind "server/db/pkg/handler/problem/find"
	"server/domains/problem/pkg/endpoint"
	"server/domains/problem/pkg/service"
	"server/kit/encode_decode"
	kitendpoint "server/kit/endpoint"
	kithandler "server/kit/handler"
)

var (
	Event = n.EProblemList
)

func Handle(
	eps endpoint.Endpoints,
	conn *rabbitmq.Connection,
	msg amqp.Delivery,
) {
	kithandler.HandleRequest(
		eps.List,
		conn,
		msg,
		decodeRequest,
		encodeResponse,
	)
}

type request struct {
	encode_decode.BaseAmqpRequest
	Data service.ListRequestData `json:"data"`
}

func decodeRequest(_ context.Context, deliv *amqp.Delivery) (interface{}, error) {
	var req request
	err := json.Unmarshal(deliv.Body, &req)
	return req.Data, err
}

func encodeResponse(_ context.Context, pub *amqp.Publishing, res interface{}) error {
	r := res.(kitendpoint.Response)
	r.Data = r.Data.(problemFind.ResponseData)
	b, err := json.Marshal(r)
	pub.Body = b
	return err
}
