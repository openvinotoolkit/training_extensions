package main

import (
	"flag"
	"log"
	"time"

	n "server/common/names"
	"server/domains/problem/cmd/service"
)

var amqpAddr = flag.String("amqpAddr", "idlp_rabbitmq:5672", "amqp service address")
var amqpUser = flag.String("amqpUser", "guest", "amqp service user")
var amqpPass = flag.String("amqpPass", "guest", "amqp service password")
var trainingPath = flag.String("trainingPath", "/training", "training folder path")
var problemPath = flag.String("problemPath", "/problem", "problem folder path")

func main() {
	flag.Parse()
	go NeverExit("PROBLEM")
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
	service.Run(n.QProblem, *amqpAddr, *amqpUser, *amqpPass, *trainingPath, *problemPath)
}
