package functions

import (
	"context"
	"fmt"

	transportamqp "github.com/go-kit/kit/transport/amqp"
	"github.com/streadway/amqp"

	longendpoint "server/kit/endpoint"
	longamqp "server/kit/transport/amqp"
)

func SendRequest(
	ctx context.Context,
	ch *amqp.Channel,
	queueName string,
	req interface{},
	enc transportamqp.EncodeRequestFunc,
	dec transportamqp.DecodeResponseFunc,
	isAutoAsk bool,
) (resp interface{}, err error) {
	ctx = context.WithValue(ctx, longamqp.ContextKeyAutoAck, isAutoAsk)
	q, err := ch.QueueDeclare(
		"",    // name
		false, // durable
		false, // delete when unused
		false, // exclusive
		false, // noWait
		nil,   // arguments
	)
	if err != nil {
		fmt.Println("QueueDeclare", err)
	}

	pub := transportamqp.NewPublisher(
		ch,
		&q,
		enc,
		dec,
		transportamqp.PublisherBefore(
			transportamqp.SetPublishKey(queueName),
		),
	)
	resp, err = pub.Endpoint()(ctx, req)
	if err != nil {
		fmt.Println("Endpoint", err)
	} else {
		fmt.Println("Endpoint", resp)
	}
	return resp, err
}

func SendRequestAsync(
	ctx context.Context,
	ch *amqp.Channel,
	queueName string,
	req interface{},
	enc transportamqp.EncodeRequestFunc,
	dec transportamqp.DecodeResponseFunc,
	isAutoAsk bool,
) chan longendpoint.Response {
	r := make(chan longendpoint.Response)
	go func() {
		defer close(r)
		resp, err := SendRequest(ctx, ch, queueName, req, enc, dec, isAutoAsk)
		r <- longendpoint.Response{Data: resp, Err: err}
	}()
	return r
}

type BaseRequest struct {
	Request string      `json:"request"`
	Data    interface{} `json:"data"`
}
