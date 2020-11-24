package service

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"

	n "server/common/names"
	"server/domains/model/pkg/endpoint"
	createFromGeneric "server/domains/model/pkg/handler/create_from_generic"
	"server/domains/model/pkg/handler/delete"
	"server/domains/model/pkg/handler/evaluate"
	fineTune "server/domains/model/pkg/handler/fine_tune"
	"server/domains/model/pkg/handler/list"
	updateFromlocal "server/domains/model/pkg/handler/update_from_local"
	"server/domains/model/pkg/service"
	"server/kit/encode_decode"
	kitutils "server/kit/utils"

	longendpoint "server/kit/endpoint"
)

func Run(serviceQueueName string, amqpAddr, amqpUser, amqpPass, trainingPath, problemPath *string) {
	amqpUrl := fmt.Sprintf("amqp://%s:%s@%s/", *amqpUser, *amqpPass, *amqpAddr)
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

	servicesQueuesNames := []string{n.QAsset, n.QDatabase, n.QProblem, n.QTrainModel, n.QModel, n.QBuild}
	kitutils.AmqpServicesQueuesDelare(conn, servicesQueuesNames)
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
		log.Println("Consume", serviceQueueName, err)
	}
	svc := service.New(conn, *problemPath, *trainingPath, getServiceMiddleware())
	eps := endpoint.New(svc, getEndpointMiddleware())

	go func() {
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
			case delete.Event:
				go delete.Handle(eps, conn, msg)
			case list.Event:
				go list.Handle(eps, conn, msg)
			case fineTune.Event:
				go fineTune.Handle(eps, conn, msg)
			case evaluate.Event:
				go evaluate.Handle(eps, conn, msg)
			}

			switch req.Request {
			case updateFromlocal.Request:
				go updateFromlocal.Handle(eps, conn, msg)
			case createFromGeneric.Request:
				go createFromGeneric.Handle(eps, conn, msg)
			}
		}
	}()
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
