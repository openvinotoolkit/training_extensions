package service

import (
	"encoding/json"
	"fmt"
	"log"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"

	n "server/common/names"
	t "server/common/types"
	kitutils "server/kit/utils"
	"server/workers/train/pkg/endpoint"
	getGpuAmount "server/workers/train/pkg/handler/get_gpu_amount"
	runCommand "server/workers/train/pkg/handler/run_commands"
	"server/workers/train/pkg/service"
)

func Run(serviceQueueName string, amqpAddr, amqpUser, amqpPass *string) {
	fmt.Println(*amqpAddr, *amqpUser, *amqpPass, serviceQueueName)
	amqpUrl := fmt.Sprintf("amqp://%s:%s@%s/", *amqpUser, *amqpPass, *amqpAddr)
	conn, err := rabbitmq.Dial(amqpUrl)
	if err != nil {
		log.Panic(err)
	}
	ch, err := conn.Channel()
	if err != nil {
		log.Panic(err)
	}
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
	err = ch.Qos(1, 0, false)
	if err != nil {
		log.Panicln("Qos", err)
	}
	svc := service.New(conn, getServiceMiddleware())
	eps := endpoint.New(svc)
	go func() {

		for msg := range msgs {
			var req t.AMQPRequestBody
			fmt.Println("Ready for new message")
			err := json.Unmarshal(msg.Body, &req)
			if err != nil {
				fmt.Println("Msg received", err)
				continue
			}
			fmt.Println(req.Request)
			switch req.Request {
			case runCommand.Request:
				runCommand.Handle(eps, conn, msg)
			case getGpuAmount.Request:
				getGpuAmount.Handle(eps, conn, msg)
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
