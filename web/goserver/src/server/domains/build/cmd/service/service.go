package service

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"

	"server/domains/build/pkg/endpoint"
	"server/domains/build/pkg/handler/create"
	createEmpty "server/domains/build/pkg/handler/create_empty"
	"server/domains/build/pkg/handler/list"
	updateAssetState "server/domains/build/pkg/handler/update_asset_state"
	updateTmps "server/domains/build/pkg/handler/update_tmps"
	"server/domains/build/pkg/service"
	"server/kit/encode_decode"
	longendpoint "server/kit/endpoint"
)

func Run(serviceQueueName, amqpAddr, amqpUser, amqpPass, problemPath string) {
	amqpUrl := fmt.Sprintf("amqp://%s:%s@%s/", amqpUser, amqpPass, amqpAddr)
	conn, err := rabbitmq.Dial(amqpUrl)
	if err != nil {
		log.Panic(err)
	}
	defer conn.Close()
	ch, err := conn.Channel()
	if err != nil {
		log.Panic(err)
	}
	defer ch.Close()

	msgs, err := ch.Consume(
		serviceQueueName,
		"",
		true,
		false,
		false,
		false,
		nil,
	)
	if err != nil {
		log.Println("Chanel Consume ", serviceQueueName, err)
	}

	svc := service.New(conn, problemPath, getServiceMiddleware())
	eps := endpoint.New(svc, getEndpointMiddleware())

	for msg := range msgs {
		var req encode_decode.BaseAmqpRequest
		err := json.Unmarshal(msg.Body, &req)
		if err != nil {
			fmt.Println(err)
			break
		} else {
			fmt.Println(req)
		}
		switch req.Event {
		case list.Event:
			go list.Handle(eps, conn, msg)
		case updateAssetState.Event:
			go updateAssetState.Handle(eps, conn, msg)
		case create.Event:
			go create.Handle(eps, conn, msg)
		}
		switch req.Request {
		case createEmpty.Request:
			go createEmpty.Handle(eps, conn, msg)
		case updateTmps.Request:
			go updateTmps.Handle(eps, conn, msg)
		}
	}

	select {}
}

func getServiceMiddleware() (mw []service.Middleware) {
	mw = []service.Middleware{}
	// Append your middleware here

	return
}

func getEndpointMiddleware() (mw map[string][]longendpoint.Middleware) {
	mw = map[string][]longendpoint.Middleware{}
	// Add you endpoint middleware here

	return
}

func failOnError(err error, msg string) {
	if err != nil {
		log.Println("%s: %s", msg, err)
	}
}
