package amqp

import (
	"context"
	"fmt"
	"time"

	"github.com/streadway/amqp"

	"server/kit/endpoint"
)

// The golang AMQP implementation requires the []byte representation of
// correlation id strings to have a maximum length of 255 bytes.
const maxCorrelationIdLength = 255

// Publisher wraps an AMQP channel and queue, and provides a method that
// implements endpoint.Endpoint.
type Publisher struct {
	ch        Channel
	q         *amqp.Queue
	enc       EncodeRequestFunc
	dec       DecodeResponseFunc
	before    []RequestFunc
	after     []PublisherResponseFunc
	deliverer LongDeliverer
	timeout   time.Duration
}

// NewPublisher constructs a usable Publisher for a single remote method.
func NewPublisher(
	ch Channel,
	q *amqp.Queue,
	enc EncodeRequestFunc,
	dec DecodeResponseFunc,
	options ...PublisherOption,
) *Publisher {
	p := &Publisher{
		ch:        ch,
		q:         q,
		enc:       enc,
		dec:       dec,
		deliverer: DefaultLongDeliverer,
		timeout:   time.Second * 0,
	}
	for _, option := range options {
		option(p)
	}
	return p
}

// PublisherOption sets an optional parameter for clients.
type PublisherOption func(*Publisher)

// PublisherBefore sets the RequestFuncs that are applied to the outgoing AMQP
// request before it's invoked.
func PublisherBefore(before ...RequestFunc) PublisherOption {
	return func(p *Publisher) { p.before = append(p.before, before...) }
}

// PublisherAfter sets the ClientResponseFuncs applied to the incoming AMQP
// request prior to it being decoded. This is useful for obtaining anything off
// of the response and adding onto the context prior to decoding.
func PublisherAfter(after ...PublisherResponseFunc) PublisherOption {
	return func(p *Publisher) { p.after = append(p.after, after...) }
}

// PublisherDeliverer sets the deliverer function that the Publisher invokes.
func PublisherDeliverer(deliverer LongDeliverer) PublisherOption {
	return func(p *Publisher) { p.deliverer = deliverer }
}

// PublisherTimeout sets the available timeout for an AMQP request.
func PublisherTimeout(timeout time.Duration) PublisherOption {
	return func(p *Publisher) { p.timeout = timeout }
}

// Endpoint returns a usable endpoint that invokes the remote endpoint.
func (p Publisher) Endpoint() endpoint.Endpoint {
	return func(ctx context.Context, request interface{}) chan endpoint.Response {
		var cancel context.CancelFunc
		if p.timeout > time.Second*0 {
			ctx, cancel = context.WithTimeout(ctx, p.timeout)
		} else {
			ctx, cancel = context.WithCancel(ctx)
		}

		pub := amqp.Publishing{
			DeliveryMode:  amqp.Persistent,
			ReplyTo:       p.q.Name,
			CorrelationId: randomString(randInt(5, maxCorrelationIdLength)),
		}

		returnChan := make(chan endpoint.Response)
		go func() {
			defer close(returnChan)
			defer p.ch.Close()
			if err := p.enc(ctx, &pub, request); err != nil {
				returnChan <- endpoint.Response{Data: nil, Err: endpoint.Error{Code: 1, Message: err.Error()}}
				return
			}

			for _, f := range p.before {
				// Affect only amqp.Publishing
				ctx = f(ctx, &pub, nil)
			}
			delivChan, err := p.deliverer(ctx, p, &pub)

			if err != nil {
				returnChan <- endpoint.Response{Data: nil, Err: endpoint.Error{Code: 1, Message: err.Error()}}
				return
			}
			for {
				deliv, ok := <-delivChan
				if ok == false {
					return
				}
				for _, f := range p.after {
					ctx = f(ctx, &deliv)
				}

				response, err := p.dec(ctx, &deliv)
				if err != nil {
					cancel()
					return
				}

				returnChan <- response.(endpoint.Response)
				if response.(endpoint.Response).IsLast == true {
					return
				}
			}
		}()
		return returnChan

	}
}

// Deliverer is invoked by the Publisher to publish the specified Publishing, and to
// retrieve the appropriate response Delivery object.
type LongDeliverer func(
	context.Context,
	Publisher,
	*amqp.Publishing,
) (<-chan amqp.Delivery, error)

// SendAndForgetDeliverer delivers the supplied publishing and
// returns a nil response.
// When using this deliverer please ensure that the supplied DecodeResponseFunc and
// PublisherResponseFunc are able to handle nil-type responses.
func SendAndForgetDeliverer(
	ctx context.Context,
	p Publisher,
	pub *amqp.Publishing,
) (*amqp.Delivery, error) {
	err := p.ch.Publish(
		getPublishExchange(ctx),
		getPublishKey(ctx),
		false, // mandatory
		false, // immediate
		*pub,
	)
	return nil, err
}

// DefaultDeliverer is a deliverer that publishes the specified Publishing
// and returns the first Delivery object with the matching correlationId.
// If the context times out while waiting for a reply, an error will be returned.
func DefaultLongDeliverer(
	ctx context.Context,
	p Publisher,
	pub *amqp.Publishing,
) (<-chan amqp.Delivery, error) {
	err := p.ch.Publish(
		getPublishExchange(ctx),
		getPublishKey(ctx),
		false, // mandatory
		false, // immediate
		*pub,
	)
	if err != nil {
		fmt.Println("LongDeliverer")
		return nil, err
	}
	autoAck := getConsumeAutoAck(ctx)

	deliv, err := p.ch.Consume(
		p.q.Name,
		"", // consumer
		autoAck,
		false, // exclusive
		false, // noLocal
		false, // noWait
		getConsumeArgs(ctx),
	)
	if err != nil {
		return nil, err
	}

	return deliv, nil

}
