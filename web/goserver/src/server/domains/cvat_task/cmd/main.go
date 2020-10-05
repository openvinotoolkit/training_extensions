package main

import (
	"flag"
	"log"
	"time"

	n "server/common/names"
	"server/domains/cvat_task/cmd/service"
)

var amqpAddr = flag.String("amqpAddr", "idlp_rabbitmq:5672", "amqp service address")
var amqpUser = flag.String("amqpUser", "guest", "amqp service user")
var amqpPass = flag.String("amqpPass", "guest", "amqp service password")

func main() {
	flag.Parse()
	go NeverExit("CVAT_TASK")
	select {}
}

func NeverExit(serviceName string) {
	defer func() {
		if v := recover(); v != nil {
			// A panic is detected.
			time.Sleep(5 * time.Second)
			log.Println(serviceName, "is crashed. Restart it now.")
			go NeverExit(serviceName) // restart
		}
	}()
	service.Run(n.QCvatTask, amqpAddr, amqpUser, amqpPass)
}
