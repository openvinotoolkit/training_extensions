package dump

import (
	"context"
	"encoding/json"
	"log"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	n "server/common/names"
	"server/domains/cvat_task/pkg/endpoint"
	"server/domains/cvat_task/pkg/service"
	"server/kit/encode_decode"
	kithandler "server/kit/handler"
)

var (
	Event = n.EAssetDumpAnnotation
)

func Handle(
	eps endpoint.Endpoints,
	conn *rabbitmq.Connection,
	msg amqp.Delivery,
) {
	kithandler.HandleRequest(
		eps.Dump,
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

type RequestData = service.DumpRequestData

func decodeRequest(_ context.Context, deliv *amqp.Delivery) (interface{}, error) {
	var resp request
	err := json.Unmarshal(deliv.Body, &resp)
	if err != nil {
		log.Print("CvatTask.Dump.decodeRequest.Unmarshal", err)
	}
	return resp.Data, err
}

type ResponseData = service.FindInFolderResponseData

func encodeResponse(_ context.Context, pub *amqp.Publishing, req interface{}) error {
	body, err := json.Marshal(req)
	if err != nil {
		log.Print("CvatTask.Dump.encodeResponse.Marshal", err)
	}
	pub.Body = body
	return err
}
