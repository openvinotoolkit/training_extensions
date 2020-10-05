package functions

import (
	"context"

	"github.com/streadway/amqp"

	transportamqp "github.com/go-kit/kit/transport/amqp"

	longendpoint "server/kit/endpoint"
	longamqp "server/kit/transport/amqp"
)

func BaseDetails(
	ctx context.Context,
	publishKey string,
	ch *amqp.Channel,
	enc transportamqp.EncodeRequestFunc,
	dec transportamqp.DecodeResponseFunc,
	reqType string,
	req DetailsRequestBody,
	respChan chan longendpoint.Response,
) (err error) {
	ctx = context.WithValue(ctx, longamqp.ContextKeyPublishKey, publishKey)
	ctx = context.WithValue(ctx, longamqp.ContextKeyAutoAck, true)
	q, err := ch.QueueDeclare(
		"",    // name
		false, // durable
		false, // delete when unused
		false, // exclusive
		false, // noWait
		nil,   // arguments
	)

	pub := transportamqp.NewPublisher(
		ch,
		&q,
		enc,
		dec,
		transportamqp.PublisherBefore(
			transportamqp.SetPublishKey(publishKey),
		),
	)
	resp, err := pub.Endpoint()(
		ctx,
		DetailsRequest{
			Data:    req,
			Request: reqType,
		},
	)
	if err != nil {
		respChan <- longendpoint.Response{Data: resp, Err: err.Error()}
	} else {
		respChan <- longendpoint.Response{Data: resp}
	}
	return
}

type BaseFindOneRequestBody struct {
	Id string `json:"id"`
}

type DetailsRequestBody struct {
	BaseFindOneRequestBody
}

type DetailsRequest struct {
	Request string             `json:"request"`
	Data    DetailsRequestBody `json:"data"`
}
