package handler

import (
	"context"
	"fmt"
	"log"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	kittransportamqp "server/kit/transport/amqp"

	"server/kit/endpoint"
)

func SendRequest(
	ctx context.Context,
	conn *rabbitmq.Connection,
	queueName string,
	req interface{},
	enc kittransportamqp.EncodeRequestFunc,
	dec kittransportamqp.DecodeResponseFunc,
	isAutoAsk bool,
) chan endpoint.Response {
	log.Println(req, "->", queueName)
	// for conn.IsClosed() {
	// 	time.Sleep(100 * time.Millisecond)
	// }
	ch, err := conn.Channel()
	if err != nil {
		log.Println("Chanel", err)
	}
	err = ch.Qos(1, 0, false)
	if err != nil {
		log.Println("Qos", err)
	}
	ctx = context.WithValue(ctx, kittransportamqp.ContextKeyAutoAck, isAutoAsk)
	q, err := ch.QueueDeclare(
		"",    // name
		false, // durable
		true,  // delete when unused
		false, // exclusive
		false, // noWait
		nil,   // arguments
	)
	if err != nil {
		fmt.Println("QueueDeclare ", queueName, err)
	}

	pub := kittransportamqp.NewPublisher(
		ch,
		&q,
		enc,
		dec,
		kittransportamqp.PublisherBefore(
			kittransportamqp.SetPublishKey(queueName),
		),
	)
	return pub.Endpoint()(ctx, req)
}

func HandleRequest(
	e endpoint.Endpoint,
	conn *rabbitmq.Connection,
	msg amqp.Delivery,
	decodeRequest kittransportamqp.DecodeRequestFunc,
	encodeResponse kittransportamqp.EncodeResponseFunc,
) {
	log.Println("ROUTING_KEY", msg.RoutingKey)
	ch, err := conn.Channel()
	if err != nil {
		log.Println("Chanel", err)
	}
	err = ch.Qos(1, 0, false)
	if err != nil {
		log.Println("Qos", err)
	}
	kittransportamqp.NewSubscriber(
		e,
		decodeRequest,
		encodeResponse,
	).ServeDelivery(ch)(&msg)

}
