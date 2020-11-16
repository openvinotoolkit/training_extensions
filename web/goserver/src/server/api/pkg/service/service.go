package service

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/gorilla/websocket"
	"github.com/sirius1024/go-amqp-reconnect/rabbitmq"
	"github.com/streadway/amqp"

	n "server/common/names"
	longendpoint "server/kit/endpoint"
	kittransportamqp "server/kit/transport/amqp"
)

var events = n.GetEvents()

type BasicProxy struct {
	WsConn            *websocket.Conn
	WsResponse        chan WSResponse
	Conn              *rabbitmq.Connection
	ServicesPubQueues map[string]*amqp.Queue // map[qName] requestQueue
}

type Proxy interface {
	WSRead()
	WSWrite()
}

type WSRequest struct {
	Event string      `json:"event"`
	Data  interface{} `json:"data,omitempty"`
}

type WSResponse struct {
	Event string      `json:"event"`
	Data  interface{} `json:"data"`
	Err   interface{} `json:"err,omitempty"`
}

func PubRequestEncode(_ context.Context, pub *amqp.Publishing, req interface{}) error {
	body, err := json.Marshal(req)
	failOnError(err, "Marshal PubRequestEncode")
	pub.Body = body
	return err
}

type PubResponse struct {
}

func PubResponseDecode(_ context.Context, deliv *amqp.Delivery) (response interface{}, err error) {
	var body longendpoint.Response
	err = json.Unmarshal(deliv.Body, &body)
	failOnError(err, "Unmarshal PubResponseDecode")
	return body, err
}

func getPublishKey(eventName string) string {
	return events[eventName]
}

func (p *BasicProxy) requestEndpoint(ctx context.Context, request WSRequest) {
	publishKey := getPublishKey(request.Event)
	ctx = context.WithValue(ctx, kittransportamqp.ContextKeyAutoAck, true)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	ch, err := p.Conn.Channel()
	if err != nil {
		log.Println("Channel", err)
	}
	defer ch.Close()
	q, err := ch.QueueDeclare(
		"",    // name
		false, // durable
		true,  // delete when unused
		false, // exclusive
		false, // noWait
		nil,   // arguments
	)
	if err != nil {
		log.Println("QueueDeclare", err)
	}
	// p.ServicesSubQueues[request.Event] = &q
	pub := kittransportamqp.NewPublisher(
		ch,
		&q, // Response queue
		PubRequestEncode,
		PubResponseDecode,
		kittransportamqp.PublisherBefore(
			kittransportamqp.SetPublishKey(publishKey),
		),
	)
	respChan := pub.Endpoint()(ctx, request)
	for {
		select {
		case res, ok := <-respChan:
			if ok == false {
				return
			}
			p.WsResponse <- WSResponse{request.Event, res.Data, res.Err}
			if res.IsLast == true {
				return
			}
		case <-ctx.Done():
			fmt.Println(ctx.Err())
			// err := p.unsubscribe(request.Event)
			if err != nil {
				fmt.Println("unsubscribe", err)
				p.WsResponse <- WSResponse{request.Event, ctx.Err(), err.Error()}
			}
			p.WsResponse <- WSResponse{request.Event, ctx.Err(), nil}

			return
		}
	}
}

func (p *BasicProxy) WSRead(ctx context.Context) {
	var request WSRequest
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	for {
		err := p.WsConn.ReadJSON(&request)
		if err != nil {
			fmt.Println("ReadJSON", err)
			break
		}
		fmt.Println("Request", request)

		if request.Event == n.EUnsubscribe {
			continue
		}

		if _, ok := events[request.Event]; !ok {
			fmt.Println("Request", request.Event, "not found")
			p.WsResponse <- WSResponse{
				request.Event,
				"Event Not Exists",
				err,
			}
			continue
		}
		go p.requestEndpoint(ctx, request)
	}
}

func (p *BasicProxy) WSWrite(_ context.Context) {
	for {
		select {
		case response := <-p.WsResponse:
			err := p.WsConn.WriteJSON(response)
			if err != nil {
				fmt.Println("Write to ws", err)
			}
		}
	}
}

// TODO: Create utls package
func failOnError(err error, msg string) {
	if err != nil {
		log.Println("%s: %s", msg, err)
	}
}
