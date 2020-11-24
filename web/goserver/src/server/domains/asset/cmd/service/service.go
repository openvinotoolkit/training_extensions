package service

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	n "server/common/names"
	"server/domains/asset/pkg/background"
	"server/kit/encode_decode"
	kitutils "server/kit/utils"
)

var (
	logger           log.Logger
	rabbitCloseError chan *amqp.Error
	conn             *rabbitmq.Connection
	ch               *amqp.Channel
)

func Run(serviceQueueName, assetRoot, amqpAddr, amqpUser, amqpPass string) {
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
	servicesQueuesNames := []string{n.QAsset, n.QDatabase}
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

	bkg := background.New(conn, getBackgroundMiddleware())
	go bkg.UpdateFromDisk(context.TODO(), assetRoot, 2*time.Second)
	func() {
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
			}

			// Request from another service
			switch req.Request {

			}
		}
	}()
	select {}
}

func getBackgroundMiddleware() (mw []background.Middleware) {
	mw = []background.Middleware{}
	return
}
