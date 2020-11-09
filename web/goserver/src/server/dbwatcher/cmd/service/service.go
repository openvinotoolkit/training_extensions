package service

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	n "server/common/names"
	t "server/common/types"
	"server/db/pkg/endpoint"
	"server/db/pkg/service"
	longendpoint "server/kit/endpoint"
	kitutils "server/kit/utils"
)

func Run(serviceQueueName string, amqpAddr, amqpUser, amqpPass, mongoAddr *string) {
	ctx := context.Background()
	mongoUrl := fmt.Sprintf("mongodb://%s", *mongoAddr)
	client, err := mongo.Connect(ctx, options.Client().ApplyURI(mongoUrl))
	if err != nil {
		log.Panic(err)
	}
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

	db := client.Database("db")

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
	svc := service.New(db, getServiceMiddleware())
	eps := endpoint.New(svc)

	go func() {

		for msg := range msgs {
			var req t.AMQPRequestBody

			err := json.Unmarshal(msg.Body, &req)
			if err != nil {
				fmt.Println("Msg recieved", err)
				continue
			}
			fmt.Println(req.Request)
			switch req.Request {
			case problemWatch.Request:
				go problemWatch.Handle(eps, conn, msg)

			default:
				log.Println("UNKNOWN REQUEST", req.Request)
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
