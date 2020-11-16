package create_from_generic

import (
	"context"
	"encoding/json"
	"log"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	n "server/common/names"
	"server/db/pkg/types"
	"server/domains/model/pkg/endpoint"
	"server/domains/model/pkg/service"
	kitendpoint "server/kit/endpoint"
	kithandler "server/kit/handler"
)

var (
	Request = n.RModelCreateFromGeneric
	Queue   = n.QModel
)

func Send(
	ctx context.Context,
	conn *rabbitmq.Connection,
	req RequestData,
) chan kitendpoint.Response {
	return kithandler.SendRequest(
		ctx,
		conn,
		Queue,
		request{
			Data:    req,
			Request: Request,
		},
		encodeRequest,
		decodeResponse,
		true,
	)
}

func Handle(
	eps endpoint.Endpoints,
	conn *rabbitmq.Connection,
	msg amqp.Delivery,
) {
	kithandler.HandleRequest(
		eps.CreateFromGeneric,
		conn,
		msg,
		decodeRequest,
		encodeResponse,
	)
}

type request struct {
	Request string      `json:"request"`
	Data    RequestData `json:"data"`
}

type RequestData = service.CreateFromGenericRequest

func encodeRequest(_ context.Context, pub *amqp.Publishing, req interface{}) (err error) {
	b, err := json.Marshal(req.(request))
	if err != nil {
		log.Println("domains.model.pkg.handler.create_from_generic.create_from_generic.encodeRequest.json.json.Marshal(req.(request))", err)
	}
	pub.Body = b
	return
}

func decodeRequest(_ context.Context, deliv *amqp.Delivery) (interface{}, error) {
	var req request
	err := json.Unmarshal(deliv.Body, &req)
	if err != nil {
		log.Println("domains.model.pkg.handler.create_from_generic.create_from_generic.decodeRequest.json.Unmarshal(deliv.Body, &req)", err)
	}
	return req.Data, err
}

type ResponseData = types.Model

func decodeResponse(_ context.Context, deliv *amqp.Delivery) (interface{}, error) {
	var resData ResponseData
	res := kitendpoint.Response{Data: &resData}
	err := json.Unmarshal(deliv.Body, &res)
	if err != nil {
		log.Println("domains.model.pkg.handler.create_from_generic.create_from_generic.decodeResponse.json.Unmarshal(deliv.Body, &res)", err)
	}
	res.Data = resData
	return res, err
}

func encodeResponse(_ context.Context, pub *amqp.Publishing, resp interface{}) error {
	b, err := json.Marshal(resp.(kitendpoint.Response))
	if err != nil {
		log.Println("domains.model.pkg.handler.create_from_generic.create_from_generic.encodeResponse.json.Marshal(resp.(kitendpoint.Response))", err)
	}
	pub.Body = b
	return err
}
