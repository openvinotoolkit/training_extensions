package service

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"

	n "server/common/names"
	"server/domains/problem/pkg/endpoint"
	"server/domains/problem/pkg/handler/create"
	"server/domains/problem/pkg/handler/delete"
	"server/domains/problem/pkg/handler/details"
	"server/domains/problem/pkg/handler/list"
	updateFromLocal "server/domains/problem/pkg/handler/update_from_local"
	"server/domains/problem/pkg/service"
	"server/kit/encode_decode"
	longendpoint "server/kit/endpoint"
	kitutils "server/kit/utils"
)

func Run(serviceQueueName, amqpAddr, amqpUser, amqpPass, trainPath, problemPath string) {
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
	servicesQueuesNames := []string{n.QAsset, n.QDatabase, n.QProblem, n.QTrainModel, n.QModel, n.QBuild}
	kitutils.AmqpServicesQueuesDelare(conn, servicesQueuesNames)
	msgs, err := ch.Consume(
		serviceQueueName,
		"",
		true,
		true, // was changed
		false,
		false,
		nil,
	)
	if err != nil {
		log.Println("Consume", serviceQueueName, err)
	}
	svc := service.New(conn, problemPath, trainPath, getServiceMiddleware())
	eps := endpoint.New(svc, getEndpointMiddleware())

	go func() {
		for msg := range msgs {
			var req encode_decode.BaseAmqpRequest
			err := json.Unmarshal(msg.Body, &req)
			if err != nil {
				fmt.Println("Msg recieved", err)
				continue
			}
			fmt.Println("Request:", req)
			// Event from UI
			switch req.Event {
			case create.Event:
				go create.Handle(eps, conn, msg)
			case delete.Event:
				go delete.Handle(eps, conn, msg)
			case details.Event:
				go details.Handle(eps, conn, msg)
			case list.Event:
				go list.Handle(eps, conn, msg)
			}

			// Request from another service
			switch req.Request {
			case updateFromLocal.Request:
				go updateFromLocal.Handle(eps, conn, msg)
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
