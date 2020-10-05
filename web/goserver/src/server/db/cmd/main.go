package main

import (
	"flag"
	"log"
	"time"

	n "server/common/names"
	"server/db/cmd/service"
)

var amqpAddr = flag.String("amqpAddr", "idlp_rabbitmq:5672", "amqp service address")
var amqpUser = flag.String("amqpUser", "guest", "amqp service user")
var amqpPass = flag.String("amqpPass", "guest", "amqp service password")
var mongoAddr = flag.String("mongoAddr", "idlp_mongo:27017", "mongodb addr")

func main() {
	flag.Parse()
	go NeverExit("DATABASE")
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
	service.Run(n.QDatabase, amqpAddr, amqpUser, amqpPass, mongoAddr)
}
