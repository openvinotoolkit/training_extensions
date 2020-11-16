package find

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
	Request = n.RDBCvatTaskFind
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
		eps.CvatTaskFind,
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

type RequestData = service.CvatTaskFindRequestData

func encodeRequest(_ context.Context, pub *amqp.Publishing, req interface{}) (err error) {
	b, err := json.Marshal(req.(request))
	if err != nil {
		log.Println("cvat_task.find.encodeRequest.Marshal", err)
	}
	pub.Body = b
	return
}

func decodeRequest(_ context.Context, deliv *amqp.Delivery) (interface{}, error) {
	var req request
	err := json.Unmarshal(deliv.Body, &req)
	if err != nil {
		log.Println("cvat_task.find.decodeRequest.Unmarshal", err)
	}
	return req.Data, err
}

type ResponseData = service.CvatTaskFindResponse

func decodeResponse(_ context.Context, deliv *amqp.Delivery) (interface{}, error) {
	var resData ResponseData
	res := kitendpoint.Response{Data: &resData}
	err := json.Unmarshal(deliv.Body, &res)
	if err != nil {
		log.Println("cvat_task.find.decodeResponse.Unmarshal(deliv.Body, &res)", err)
		res.Err = kitendpoint.Error{Code: 1, Message: err.Error()}
	}
	res.Data = resData
	return res, err
}

func encodeResponse(_ context.Context, pub *amqp.Publishing, resp interface{}) error {
	b, err := json.Marshal(resp.(kitendpoint.Response))
	if err != nil {
		log.Println("cvat_task.find.encodeResponse.Marshal", err)
	}
	pub.Body = b
	return err
}
