package service

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	n "server/common/names"
	"server/domains/cvat_task/pkg/endpoint"
	"server/domains/cvat_task/pkg/handler/dump"
	findInFolder "server/domains/cvat_task/pkg/handler/find_in_folder"
	"server/domains/cvat_task/pkg/handler/setup"
	"server/domains/cvat_task/pkg/service"
	"server/kit/encode_decode"
	longendpoint "server/kit/endpoint"
	kitutils "server/kit/utils"
)

var (
	logger           log.Logger
	rabbitCloseError chan *amqp.Error
	conn             *rabbitmq.Connection
	ch               *amqp.Channel
)

func Run(serviceQueueName string, amqpAddr, amqpUser, amqpPass *string) {
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

	servicesQueuesNames := []string{n.QCvatTask, n.QDatabase}
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
	svc := service.New(conn, getServiceMiddleware())
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
			case dump.Event:
				go dump.Handle(eps, conn, msg)
			case findInFolder.Event:
				go findInFolder.Handle(eps, conn, msg)
			case setup.Event:
				go setup.Handle(eps, conn, msg)
			}

			switch req.Request {
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
