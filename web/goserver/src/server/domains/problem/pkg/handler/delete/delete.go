package delete

import (
	"context"
	"encoding/json"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	n "server/common/names"
	"server/domains/problem/pkg/endpoint"
	"server/domains/problem/pkg/service"
	"server/kit/encode_decode"
	kithandler "server/kit/handler"
)

var (
	Event = n.EProblemDelete
)

func Handle(
	eps endpoint.Endpoints,
	conn *rabbitmq.Connection,
	msg amqp.Delivery,
) {
	kithandler.HandleRequest(
		eps.Delete,
		conn,
		msg,
		decodeRequest,
		encodeResponse,
	)
}

func decodeRequest(_ context.Context, deliv *amqp.Delivery) (interface{}, error) {
	var resp request
	err := json.Unmarshal(deliv.Body, &resp)
	return resp.Data, err
}

func encodeResponse(_ context.Context, pub *amqp.Publishing, req interface{}) error {
	body, err := json.Marshal(req)
	pub.Body = body
	return err
}

type request struct {
	encode_decode.BaseAmqpRequest
	Data service.DeleteRequestData `json:"data"`
}
