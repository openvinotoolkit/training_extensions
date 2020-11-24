package list

import (
	"context"
	"encoding/json"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	n "server/common/names"
	modelFind "server/db/pkg/handler/model/find"
	"server/domains/model/pkg/service"
	kited "server/kit/encode_decode"
	kitendpoint "server/kit/endpoint"
	kithandler "server/kit/handler"

	"server/domains/model/pkg/endpoint"
)

var Event = n.EModelList

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

type RequestData = service.ListRequestData

type request struct {
	kited.BaseAmqpRequest
	Data RequestData `json:"data"`
}

func decodeRequest(_ context.Context, deliv *amqp.Delivery) (interface{}, error) {
	var req request
	err := json.Unmarshal(deliv.Body, &req)
	return req.Data, err
}

type ResponseData = modelFind.ResponseData

func encodeResponse(_ context.Context, pub *amqp.Publishing, res interface{}) error {
	r := res.(kitendpoint.Response)
	r.Data = r.Data.(ResponseData)
	b, err := json.Marshal(r)
	pub.Body = b
	return err
}
