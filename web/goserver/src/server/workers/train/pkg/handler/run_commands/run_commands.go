package run_commands

import (
	"context"
	"encoding/json"
	"log"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	n "server/common/names"
	kitendpoint "server/kit/endpoint"
	kithandler "server/kit/handler"
	"server/workers/train/pkg/endpoint"
	"server/workers/train/pkg/service"
)

var (
	Request = n.RTrainModelRunCommands
	Queue   = n.QTrainModel
)

type RequestData = service.RunCommandsRequestData
type ResponseData = interface{}

func Handle(
	eps endpoint.Endpoints,
	conn *rabbitmq.Connection,
	msg amqp.Delivery,
) {
	kithandler.HandleRequest(
		eps.RunCommands,
		conn,
		msg,
		decodeRequest,
		encodeResponse,
	)
	log.Println("RUN COMMAND FINISHED")
}

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
		false,
	)
}

type request struct {
	Request string      `json:"request"`
	Data    RequestData `json:"data"`
}

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

func decodeResponse(_ context.Context, deliv *amqp.Delivery) (interface{}, error) {
	var res kitendpoint.Response
	var resData ResponseData
	var resErr kitendpoint.Error
	err := json.Unmarshal(deliv.Body, &res)
	bData, err := json.Marshal(res.Data)
	err = json.Unmarshal(bData, &resData)
	res.Data = resData
	bErr, err := json.Marshal(res.Err)
	err = json.Unmarshal(bErr, &resErr)
	res.Err = resErr
	return res, err
}

func encodeResponse(_ context.Context, pub *amqp.Publishing, resp interface{}) error {
	b, err := json.Marshal(resp.(kitendpoint.Response))
	pub.Body = b
	return err
}
