package evaluate

import (
	"context"
	"encoding/json"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	n "server/common/names"
	"server/domains/model/pkg/endpoint"
	"server/domains/model/pkg/service"
	kited "server/kit/encode_decode"
	kithandler "server/kit/handler"
)

var Event = n.EModelEvaluate

func Handle(
	eps endpoint.Endpoints,
	conn *rabbitmq.Connection,
	msg amqp.Delivery,
) {
	kithandler.HandleRequest(
		eps.Evaluate,
		conn,
		msg,
		decodeRequest,
		encodeResponse,
	)
}

type RequestData = service.EvaluateRequest

type request struct {
	kited.BaseAmqpRequest
	Data RequestData `json:"data"`
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
