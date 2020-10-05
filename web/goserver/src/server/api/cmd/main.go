package main

import (
	"flag"
	"log"
	"time"

	"server/api/cmd/service"
)

var httpAddr = flag.String("httpAddr", "idlp_api:8888", "http service address")
var amqpAddr = flag.String("amqpAddr", "idlp_rabbitmq:5672", "amqp service address")
var amqpUser = flag.String("amqpUser", "guest", "amqp service user")
var amqpPass = flag.String("amqpPass", "guest", "amqp service password")
var oteProblemsPath = flag.String("oteProblemsPath", "/ote/pytorch_toolkit", "problem folder path")

func main() {
	flag.Parse()
	go NeverExit("API")
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
	service.Run(*httpAddr, *amqpUser, *amqpPass, *amqpAddr, *oteProblemsPath)
}
