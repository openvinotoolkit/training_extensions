package service

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
	"log"
	cvatTaskFind "server/db/pkg/handler/cvat_task/find"

	t "server/common/types"
	"server/db/pkg/endpoint"
	"server/db/pkg/service"
)

func Run(serviceQueueName string, amqpAddr, amqpUser, amqpPass, mongoAddr *string) {
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
	ctx := context.Background()
	mongoUrl := fmt.Sprintf("mongodb://%s", *mongoAddr)
	client, err := mongo.Connect(ctx, options.Client().ApplyURI(mongoUrl))
	failOnError(err, "Mongo Connect")
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
			case cvatTaskFind.Request:
				go cvatTaskFind.Handle(eps, conn, msg)
			default:
				log.Println("UNKNOWN REQUEST", req.Request)
			}
		}
	}()
	select {}
}

func failOnError(err error, msg string) {
	if err != nil {
		log.Println("%s: %s", msg, err)
	}
}

func getServiceMiddleware() (mw []service.Middleware) {
	mw = []service.Middleware{}
	// Append your middleware here

	return
}
