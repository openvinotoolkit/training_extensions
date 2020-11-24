package delete

import (
	"context"
	"encoding/json"
	"log"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	n "server/common/names"
	"server/db/pkg/endpoint"
	"server/db/pkg/service"
	kitendpoint "server/kit/endpoint"
	kithandler "server/kit/handler"
)

var (
	Request = n.RDBModelDelete
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
			Request: Request,
			Data:    req,
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
		eps.ModelDelete,
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

type RequestData = service.ModelDeleteRequestData

func encodeRequest(_ context.Context, pub *amqp.Publishing, req interface{}) (err error) {
	b, err := json.Marshal(req.(request))
	if err != nil {
		log.Println("model.find.encodeRequest.Marshal", err)
	}
	pub.Body = b
	return
}

func decodeRequest(_ context.Context, deliv *amqp.Delivery) (interface{}, error) {
	var req request
	err := json.Unmarshal(deliv.Body, &req)
	return req.Data, err
}

type ResponseData = service.ModelDeleteResponseData

func decodeResponse(_ context.Context, deliv *amqp.Delivery) (interface{}, error) {
	var res kitendpoint.Response
	var resData ResponseData
	err := json.Unmarshal(deliv.Body, &res)
	b, err := json.Marshal(res.Data)
	err = json.Unmarshal(b, &resData)
	res.Data = resData
	return res, err
}

func encodeResponse(_ context.Context, pub *amqp.Publishing, resp interface{}) error {
	b, err := json.Marshal(resp.(kitendpoint.Response))
	pub.Body = b
	return err
}
