package find_one

import (
	"context"
	"encoding/json"
	"log"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	n "server/common/names"
	"server/db/pkg/endpoint"
	"server/db/pkg/service"
	"server/db/pkg/types"
	kitendpoint "server/kit/endpoint"
	kithandler "server/kit/handler"
)

var (
	Request = n.RDBBuildFindOne
	Queue   = n.QDatabase
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
		eps.BuildFindOne,
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

type RequestData = service.BuildFindOneRequestData

func encodeRequest(_ context.Context, pub *amqp.Publishing, req interface{}) (err error) {
	b, err := json.Marshal(req.(request))
	pub.Body = b
	return
}

func decodeRequest(_ context.Context, deliv *amqp.Delivery) (interface{}, error) {
	var req request
	err := json.Unmarshal(deliv.Body, &req)
	return req.Data, err
}

type ResponseData = types.Build

func decodeResponse(_ context.Context, deliv *amqp.Delivery) (interface{}, error) {
	var resData ResponseData
	res := kitendpoint.Response{Data: &resData}
	err := json.Unmarshal(deliv.Body, &res)
	res.Err = kitendpoint.Error{Code: 0}
	if err != nil {
		log.Println("build.find_one.decodeResponse.Unmarshal(deliv.Body, &res)", err)
		res.Err = kitendpoint.Error{Code: 1, Message: err.Error()}
	}
	res.Data = resData
	return res, err
}

func encodeResponse(_ context.Context, pub *amqp.Publishing, resp interface{}) error {
	b, err := json.Marshal(resp.(kitendpoint.Response))
	pub.Body = b
	return err
}
