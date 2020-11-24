package list

import (
	"context"
	"encoding/json"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	n "server/common/names"
	buildFind "server/db/pkg/handler/build/find"
	"server/domains/build/pkg/endpoint"
	"server/domains/build/pkg/service"
	"server/kit/encode_decode"
	kitendpoint "server/kit/endpoint"
	kithandler "server/kit/handler"
)

var (
	Event = n.EBuildList
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
	Data RequestData `json:"data"`
}

type RequestData = service.ListRequestData

func decodeRequest(_ context.Context, deliv *amqp.Delivery) (interface{}, error) {
	var req request
	err := json.Unmarshal(deliv.Body, &req)
	return req.Data, err
}

type ResponseData = buildFind.ResponseData

func encodeResponse(_ context.Context, pub *amqp.Publishing, res interface{}) error {
	r := res.(kitendpoint.Response)
	r.Data = r.Data.(ResponseData)
	b, err := json.Marshal(r)
	pub.Body = b
	return err
}
